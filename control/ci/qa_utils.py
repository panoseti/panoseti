import asyncio
import json
import os
import re
import sys
import time
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# We keep Pydantic imports at module level for model definitions, 
# but Pydantic v2 is generally fast.
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    pass

# ── ANSI Colors ──────────────────────────────────────────────────────────────

PALETTE = [
    "deep_sky_blue1",
    "spring_green1",
    "orange1",
    "magenta1",
    "medium_purple1",
    "cyan1",
]

# ── Configuration Models ──────────────────────────────────────────────────────

class SuiteConfig(BaseModel):
    name: str = ""
    description: str = ""
    type: Literal["test", "lint"] = "test"
    environment: str | None = None
    requires_docker: bool = False
    compose_file: str | None = None
    profiles: list[str] = Field(default_factory=list)
    service: str | None = None
    test_dir: str | None = None
    parallel: bool = False
    pytest_args: list[str] = Field(default_factory=list)
    pre_run: str | None = None
    tasks: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)

class EnvironmentConfig(BaseModel):
    """Maps to [environments.X] in qa.toml."""
    config_dir: str # Path to PANOSETI JSON configs for this env
    compose_file: str # Compose file for this environment

class QAConfig(BaseModel):
    settings: dict[str, Any] = Field(default_factory=dict)
    environments: dict[str, EnvironmentConfig] = Field(default_factory=dict)
    suites: dict[str, SuiteConfig]

    @field_validator("suites", mode="before")
    @classmethod
    def inject_names(cls, v: Any) -> Any:
        if isinstance(v, dict):
            for name, config in v.items():
                if isinstance(config, dict) and "name" not in config:
                    config["name"] = name
        return v

# ── Core Types ────────────────────────────────────────────────────────────────

class Result:
    __slots__ = ("code", "elapsed", "name", "stats", "stdout")
    def __init__(self, name: str, code: int, elapsed: float, stats: dict[str, int] | None = None, stdout: str = "") -> None:
        self.name    = name
        self.code    = code
        self.elapsed = elapsed
        self.stats   = stats or {}
        self.stdout  = stdout
    @property
    def ok(self) -> bool: return self.code == 0

class SSHTunnel:
    """Manages a background SSH tunnel for Unix Domain Sockets."""
    def __init__(self, ssh_args: str, remote_socket: str, local_socket: str = "/tmp/pseti_tunnel.sock"):
        self.ssh_args = ssh_args
        self.remote_socket = remote_socket
        self.local_socket = local_socket
        self.proc: Any | None = None

    def __enter__(self):
        import subprocess
        # Remove stale local socket
        if os.path.exists(self.local_socket):
            os.unlink(self.local_socket)
        
        # Start tunnel in background: -n (no stdin), -N (no command), -L (local forward)
        cmd = ["ssh", "-n", "-N", "-L", f"{self.local_socket}:{self.remote_socket}"]
        import shlex
        cmd.extend(shlex.split(self.ssh_args))
        
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for socket to appear
        start = time.monotonic()
        while time.monotonic() - start < 5:
            if os.path.exists(self.local_socket):
                return self.local_socket
            time.sleep(0.1)
        
        self.stop()
        raise TimeoutError(f"SSH Tunnel to {self.remote_socket} failed to establish.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        import contextlib
        if self.proc:
            self.proc.terminate()
            with contextlib.suppress(Exception):
                self.proc.wait(timeout=1)
            self.proc = None
        if os.path.exists(self.local_socket):
            with contextlib.suppress(Exception):
                os.unlink(self.local_socket)

CONTROL_ROOT = Path(__file__).parent.parent.resolve()
QA_TOML_PATH = CONTROL_ROOT / "ci" / "qa.toml"
ENV_CI_PATH  = CONTROL_ROOT / "ci" / ".env.ci"

# ── Runner Implementation ─────────────────────────────────────────────────────

class TestRunner:
    def __init__(self, config_path: Path):
        try:
            with open(config_path, "rb") as fh:
                raw_cfg = tomllib.load(fh)
                self.cfg = QAConfig.model_validate(raw_cfg)
        except Exception as e:
            from rich.console import Console
            Console().print(f"[red]Error loading {config_path}: {e}[/red]")
            sys.exit(1)
        
        self.no_teardown = False
        self.no_build = False
        self.container_tool = "docker"
        self.default_parallel = self.cfg.settings.get("default_parallel", 4)
        self.project_prefix = self.cfg.settings.get("project_prefix", "pseti")

    async def run_suite(self, suite_name: str, jobs: int | None = None, extra_args: list[str] | None = None) -> bool:
        if suite_name not in self.cfg.suites:
            from rich.console import Console
            Console().print(f"[red]Unknown suite: {suite_name}[/red]")
            return False
        
        suite = self.cfg.suites[suite_name]
        project_name = f"{self.project_prefix}-{suite.name}"
        
        results: list[Result] = []
        try:
            if suite.requires_docker:
                await self._setup_docker(suite, project_name)
            
            if suite.type == "test":
                results = await self._run_test_suite(suite, project_name, jobs, extra_args)
            elif suite.type == "lint":
                results = await self._run_lint_suite(suite, project_name, extra_args)
                
        finally:
            if suite.requires_docker and not self.no_teardown:
                await self._teardown_docker(suite, project_name)
        
        self._print_summary(results)
        return all(r.ok for r in results)

    async def cleanup_all(self):
        self._header("GLOBAL CLEANUP")
        for _name, suite in self.cfg.suites.items():
            if suite.requires_docker:
                project_name = f"{self.project_prefix}-{suite.name}"
                from rich.console import Console
                Console().print(f"[dim]Cleaning up {project_name}...[/dim]")
                await self._teardown_docker(suite, project_name, quiet=True)
        from rich.console import Console
        Console().print("[green]Cleanup complete.[/green]")

    async def build_all(self):
        self._header("REBUILDING IMAGES")
        processed_files = set()
        for suite in self.cfg.suites.values():
            if suite.requires_docker and suite.compose_file not in processed_files:
                project_name = f"{self.project_prefix}-build"
                cmd = f"{self.container_tool} compose --env-file {ENV_CI_PATH} -f {CONTROL_ROOT}/{suite.compose_file} build"
                await self._run_cmd(cmd, env={"COMPOSE_PROJECT_NAME": project_name})
                processed_files.add(suite.compose_file)

    # ── Internal Helpers ──────────────────────────────────────────────────────

    async def _setup_docker(self, suite: SuiteConfig, project_name: str):
        self._header(f"SETUP: {suite.name.upper()}")
        profile_str = " ".join([f"--profile {p}" for p in suite.profiles])
        
        compose_file = suite.compose_file
        if not compose_file and suite.environment:
            env_cfg = self.cfg.environments.get(suite.environment)
            if env_cfg:
                compose_file = env_cfg.compose_file
        
        if not compose_file:
             from rich.console import Console
             Console().print(f"[red]Error: No compose file defined for suite {suite.name}[/red]")
             sys.exit(1)

        build_flag = " --no-build" if self.no_build else ""
        cmd = f"{self.container_tool} compose --env-file {ENV_CI_PATH} -f {CONTROL_ROOT}/{compose_file} {profile_str} up -d{build_flag}"
        res = await self._run_cmd(cmd, env={"COMPOSE_PROJECT_NAME": project_name})
        if not res.ok:
            from rich.console import Console
            Console().print(f"[red]Failed to start container stack for {suite.name}[/red]")
            sys.exit(1)
            
        if suite.pre_run:
            from rich.console import Console
            Console().print(f"[dim]Running pre-run command for {suite.name}...[/dim]")
            pre_cmd = f"{self.container_tool} compose --env-file {ENV_CI_PATH} -f {CONTROL_ROOT}/{compose_file} {profile_str} exec -T {suite.service} /bin/sh -c '{suite.pre_run}'"
            await self._run_cmd(pre_cmd, env={"COMPOSE_PROJECT_NAME": project_name})

    async def _teardown_docker(self, suite: SuiteConfig, project_name: str, quiet: bool = False):
        if not quiet:
            self._header(f"TEARDOWN: {suite.name.upper()}")
        profile_str = " ".join([f"--profile {p}" for p in suite.profiles])
        
        compose_file = suite.compose_file
        if not compose_file and suite.environment:
            env_cfg = self.cfg.environments.get(suite.environment)
            if env_cfg:
                compose_file = env_cfg.compose_file
        
        if not compose_file:
            return

        cmd = f"{self.container_tool} compose --env-file {ENV_CI_PATH} -f {CONTROL_ROOT}/{compose_file} {profile_str} down -v --remove-orphans"
        await self._run_cmd(cmd, env={"COMPOSE_PROJECT_NAME": project_name}, quiet=quiet)

    async def _run_test_suite(self, suite: SuiteConfig, project_name: str, jobs: int | None, extra_args: list[str] | None) -> list[Result]:
        self._header(f"TESTING: {suite.name.upper()}")
        
        p = jobs or self.default_parallel
        args = suite.pytest_args + (extra_args or [])
        args_str = " ".join(args)
        
        pytest_cmd = f"pytest {suite.test_dir} -v --color=yes"
        if suite.parallel:
            pytest_cmd += f" -n {p}"
        if args_str:
            pytest_cmd += f" {args_str}"
            
        env_str = " ".join([f"-e {k}={v}" for k, v in suite.env.items()])
        profile_str = " ".join([f"--profile {p}" for p in suite.profiles])
        
        compose_file = suite.compose_file
        if not compose_file and suite.environment:
            env_cfg = self.cfg.environments.get(suite.environment)
            if env_cfg:
                compose_file = env_cfg.compose_file

        cmd = f"{self.container_tool} compose --env-file {ENV_CI_PATH} -f {CONTROL_ROOT}/{compose_file} {profile_str} exec -T {env_str} {suite.service} {pytest_cmd}"
        
        lock = asyncio.Lock()
        res = await self._stream(f"test.{suite.name}", cmd, lock, env={"COMPOSE_PROJECT_NAME": project_name})
        return [res]

    async def _run_lint_suite(self, suite: SuiteConfig, project_name: str, extra_args: list[str] | None) -> list[Result]:
        self._header(f"LINTING: {suite.name.upper()}")
        
        extra_str = " ".join(extra_args or [])
        profile_str = " ".join([f"--profile {p}" for p in suite.profiles])
        
        compose_file = suite.compose_file
        if not compose_file and suite.environment:
            env_cfg = self.cfg.environments.get(suite.environment)
            if env_cfg:
                compose_file = env_cfg.compose_file

        lock = asyncio.Lock()
        
        async def run_task(name: str, task_cmd: str):
            cmd = f"{self.container_tool} compose --env-file {ENV_CI_PATH} -f {CONTROL_ROOT}/{compose_file} {profile_str} exec -T {suite.service} {task_cmd} {extra_str}"
            tag_text = f"[{name}] "
            return await self._stream(f"lint.{name}", cmd, lock, tag=tag_text, env={"COMPOSE_PROJECT_NAME": project_name})

        results = await asyncio.gather(*[run_task(n, c) for n, c in suite.tasks.items()])
        return list(results)

    async def _run_cmd(self, cmd: str, env: dict[str, str] | None = None, quiet: bool = False, capture: bool = False) -> Result:
        start = time.monotonic()
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
            
        stdout_dest = asyncio.subprocess.PIPE if capture else (asyncio.subprocess.DEVNULL if quiet else None)
        
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=stdout_dest,
            stderr=asyncio.subprocess.DEVNULL if quiet else None,
            env=full_env
        )
        
        stdout_data = ""
        if capture:
            stdout_bytes, _ = await proc.communicate()
            stdout_data = stdout_bytes.decode("utf-8", errors="replace")
        else:
            await proc.wait()
            
        return Result("cmd", proc.returncode or 0, time.monotonic() - start, stdout=stdout_data)

    async def _stream(self, name: str, cmd: str, lock: asyncio.Lock, tag: str = "", env: dict[str, str] | None = None) -> Result:
        start = time.monotonic()
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
            
        proc  = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=full_env
        )
        assert proc.stdout is not None

        stats = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
        is_parallel = "-n" in cmd
        has_json_metrics = False
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            plain_line = ansi_escape.sub('', line)
            
            if "TEST_METRICS_JSON: " in plain_line:
                try:
                    json_str = plain_line.split("TEST_METRICS_JSON: ")[1]
                    summary = json.loads(json_str)
                    stats["passed"] = summary.get("passed", 0) + summary.get("xpass", 0)
                    stats["failed"] = summary.get("failed", 0)
                    stats["skipped"] = summary.get("skipped", 0) + summary.get("xfail", 0)
                    stats["error"] = summary.get("error", 0)
                    has_json_metrics = True
                except Exception:
                    pass

            upper_line = plain_line.upper()
            if not has_json_metrics:
                is_result = (plain_line.startswith("[gw") and any(kw in upper_line for kw in [" PASSED", " FAILED", " SKIPPED", " ERROR", " XFAIL", " XPASS"])) or \
                            ("::" in plain_line and any(upper_line.endswith(kw) or f"{kw} [" in upper_line for kw in ["PASSED", "FAILED", "SKIPPED", "ERROR", "XFAIL", "XPASS"]))
                if is_result:
                    if "PASSED" in upper_line or "XPASS" in upper_line:
                        stats["passed"] += 1
                    elif "FAILED" in upper_line:
                        stats["failed"] += 1
                    elif "SKIPPED" in upper_line or "XFAIL" in upper_line:
                        stats["skipped"] += 1
                    elif "ERROR" in upper_line:
                        stats["error"] += 1

            if line.startswith("[gw"):
                end = line.find("]")
                if end != -1:
                    wid = line[:end+1]
                    line = f"{wid}{line[end+1:]}"
            elif is_parallel and "::" in plain_line and not plain_line.startswith("["):
                if not any(kw in upper_line for kw in ["PASSED", "FAILED", "SKIPPED", "ERROR", "XFAIL", "XPASS"]):
                    continue

            async with lock:
                print(f"{tag}{line}", flush=True)

        await proc.wait()
        return Result(name, proc.returncode or 0, time.monotonic() - start, stats)

    def _header(self, title: str):
        from rich.console import Console
        from rich.panel import Panel
        Console().print(Panel.fit(f"[bold yellow]{title}[/bold yellow]", border_style="yellow"))

    def _print_summary(self, results: list[Result]):
        if not results:
            return
        from rich.console import Console
        console = Console()
        width = max(len(r.name) for r in results)
        console.print("\n[bold]Execution Summary[/bold]")
        for r in results:
            icon = "[green]✓[/green]" if r.ok else "[red]✗[/red]"
            status = "[green]passed[/green]" if r.ok else "[red]FAILED[/red]"
            console.print(f"  {icon}  [cyan]{r.name.ljust(width)}[/cyan]  {status}  [dim]{r.elapsed:.1f}s[/dim]")
        
        test_res = [r for r in results if r.name.startswith("test.") and any(r.stats.values())]
        if test_res:
            self._print_metrics(test_res)

    def _print_metrics(self, test_results: list[Result]):
        from rich.console import Console
        console = Console()
        console.print("\n[bold]Test Metrics[/bold]")
        header = f"  {'Suite':<20} {'Passed':>8} {'Failed':>8} {'Skipped':>8} {'Error':>8} {'Total':>8}"
        bar = "  " + "─" * (len(header) - 2)
        console.print(f"[dim]{bar}[/dim]")
        console.print(f"[bold yellow]{header}[/bold yellow]")
        console.print(f"[dim]{bar}[/dim]")
        totals = {"p": 0, "f": 0, "s": 0, "e": 0, "t": 0}
        for r in test_results:
            s = r.stats
            p, f, sk, e = s.get("passed", 0), s.get("failed", 0), s.get("skipped", 0), s.get("error", 0)
            t = p + f + sk + e
            
            p_str = str(p).rjust(8)
            f_str = str(f).rjust(8)
            sk_str = str(sk).rjust(8)
            e_str = str(e).rjust(8)
            t_str = str(t).rjust(8)

            console.print(
                f"  {r.name:<20} "
                f"[green]{p_str}[/green] "
                f"[red]{f_str}[/red] "
                f"[yellow]{sk_str}[/yellow] "
                f"[red]{e_str}[/red] "
                f"{t_str}"
            )
            totals["p"] += p
            totals["f"] += f
            totals["s"] += sk
            totals["e"] += e
            totals["t"] += t
        console.print(f"[dim]{bar}[/dim]")
        
        tp_str = str(totals['p']).rjust(8)
        tf_str = str(totals['f']).rjust(8)
        ts_str = str(totals['s']).rjust(8)
        te_str = str(totals['e']).rjust(8)
        tt_str = str(totals['t']).rjust(8)

        console.print(
            f"  {'Total':<20} "
            f"[green]{tp_str}[/green] "
            f"[red]{tf_str}[/red] "
            f"[yellow]{ts_str}[/yellow] "
            f"[red]{te_str}[/red] "
            f"{tt_str}\n"
        )
