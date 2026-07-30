import asyncio
import contextlib
import os
import random
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

from ci.shared.qa_models import QAConfig, SuiteConfig
from ci.shared.stream import Result, stream_test_output
from control.utils.paths import PanoPaths

# Magic paths
CONTROL_ROOT = PanoPaths.base_dir()
CI_ROOT = CONTROL_ROOT / "src" / "ci"
QA_TOML_PATH = CI_ROOT / "qa.sw.toml"
ENV_CI_PATH  = CI_ROOT / ".env.ci"

# ── ANSI Colors ──────────────────────────────────────────────────────────────

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


# ── Runner Implementation ─────────────────────────────────────────────────────

class TestRunner:
    def __init__(self, config_path: Path):
        try:
            with open(config_path, "rb") as fh:
                raw_cfg = tomllib.load(fh)
                self.cfg = QAConfig.model_validate(raw_cfg)
        except Exception as e:
            from rich.console import Console
            Console().print(f"[red]Error loading '{config_path}': {e}[/red]")
            sys.exit(1)
        
        self.no_teardown = False
        self.no_build = False
        self.dev_mode = False
        self.container_tool = "docker"
        self.default_parallel = self.cfg.settings.get("default_parallel", 4)
        self.project_prefix = self.cfg.settings.get("project_prefix", "pseti")
        self._temp_envs: dict[str, tuple[Path, dict[str, str]]] = {}
        self.all_results: list[Result] = []

    async def run_suite(self, suite_name: str, jobs: int | None = None, target: str | None = None, extra_args: list[str] | None = None) -> bool:
        if suite_name not in self.cfg.suites:
            from rich.console import Console
            Console().print(f"[red]Unknown suite: {suite_name}[/red]")
            return False

        suite = self.cfg.suites[suite_name]
        project_name = f"{self.project_prefix}-{suite_name}"
        
        # 1. Setup
        if suite.requires_docker:
            await self._setup_docker(suite, project_name)
        
        # 2. Run
        results = []
        try:
            if suite.type == "lint":
                results = await self._run_lint_suite(suite, project_name, target, extra_args)
            else:
                results = await self._run_test_suite(suite, project_name, jobs, extra_args)
        finally:
            # 3. Teardown
            if suite.requires_docker and not self.no_teardown:
                await self._teardown_docker(suite, project_name)
            elif not self.no_teardown:
                # Even if suite doesn't require docker, it might have spawned 
                # testcontainers (e.g. sw2 fleet). Prune them.
                await self._cleanup_orphaned_networks(project_name, quiet=True)
            
            # Clean up temp env file
            if suite_name in self._temp_envs:
                self._temp_envs[suite_name][0].unlink(missing_ok=True)
                del self._temp_envs[suite_name]

        self.all_results.extend(results)
        success = all(r.ok for r in results)
        if not success:
            from rich.console import Console
            c = Console()
            for r in results:
                if not r.ok:
                    c.print(f"[red]Suite {suite_name} result {r.name} failed with code {r.code}[/red]")
                    c.print(f"[dim]Stats: {r.stats}[/dim]")
        return success

    async def build_images(self, suite_name: str | None = None):
        """Pre-build all images used in the suite(s)."""
        self._header("BUILDING IMAGES")
        
        processed_files = set()
        suites_to_build = [self.cfg.suites[suite_name]] if suite_name else self.cfg.suites.values()
        
        for suite in suites_to_build:
            from rich.console import Console
            c = Console()
            if not suite.requires_docker:
                c.print(f"[dim]Skipping suite {suite.name} (no docker)[/dim]")
                continue
                
            compose_file = suite.compose_file
            if not compose_file and suite.environment:
                env_cfg = self.cfg.environments.get(suite.environment)
                if env_cfg:
                    compose_file = env_cfg.compose_file
                else:
                    c.print(f"[yellow]Warning: environment {suite.environment} not found for suite {suite.name}[/yellow]")
            
            if not compose_file:
                c.print(f"[yellow]Warning: no compose file found for suite {suite.name}[/yellow]")
                continue

            if compose_file in processed_files:
                continue

            c.print(f"[dim]Processing compose file for build: {compose_file}[/dim]")
            project_name = f"{self.project_prefix}-build"
            
            # Inject host UID/GID for build-time injection
            build_env = {
                "COMPOSE_PROJECT_NAME": project_name,
                "LOCAL_UID": str(os.getuid()),
                "LOCAL_GID": str(os.getgid())
            }
            
            # Use 'compose config' to find all services in this file
            config_cmd = f"{self.container_tool} compose --env-file {ENV_CI_PATH} -f {CI_ROOT}/{compose_file} config --services"
            res = await self._run_cmd(config_cmd, env=build_env, capture=True)
            
            if res.ok and res.stdout:
                services = res.stdout.strip().split("\n")
                for service in services:
                    from rich.console import Console
                    Console().print(f"[cyan]Building service: {service} (from {compose_file})...[/cyan]")
                    cmd = f"{self.container_tool} compose --env-file {ENV_CI_PATH} -f {CI_ROOT}/{compose_file} build {service}"
                    await self._run_cmd(cmd, env=build_env)
            else:
                # Fallback to full build if services couldn't be parsed
                cmd = f"{self.container_tool} compose --env-file {ENV_CI_PATH} -f {CI_ROOT}/{compose_file} build"
                await self._run_cmd(cmd, env=build_env)
                
            processed_files.add(compose_file)

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _generate_dynamic_env(self, suite: SuiteConfig) -> tuple[Path, dict[str, str]]:
        """Generates a temporary .env file with non-overlapping subnets."""
        # Use random prefixes to avoid collisions, but respect suite-specific overrides
        head_prefix = suite.env.get('HEAD_NET_PREFIX')
        if not head_prefix:
            x = random.randint(100, 200)
            y = random.randint(1, 250)
            head_prefix = f'10.{x}.{y}'
            
        daq_prefix = suite.env.get('DAQ_NET_PREFIX')
        if not daq_prefix:
            daq_prefix = f'192.168.{random.randint(0, 250)}'
            
        quabo_prefix = suite.env.get('QUABO_NET_PREFIX')
        if not quabo_prefix:
            quabo_prefix = f'192.168.{random.randint(0, 250)}'
        
        expanded_env = {
            "HEAD_NET_PREFIX": head_prefix,
            "DAQ_NET_PREFIX": daq_prefix,
            "QUABO_NET_PREFIX": quabo_prefix,
            "HEAD_NET_HEADNODE": f"{head_prefix}.22",
            "HEAD_NET_REDIS": f"{head_prefix}.20",
            "HEAD_NET_LOKI": f"{head_prefix}.21",
            "HEAD_NET_GATEWAY": f"{head_prefix}.254",
            "HEAD_NET_TESTER": f"{head_prefix}.5",
            "HEAD_NET_DAQNODE_1": f"{head_prefix}.10",
            "HEAD_NET_DAQNODE_2": f"{head_prefix}.11",
            "DAQ_NET_DAQNODE_1": f"{daq_prefix}.10",
            "DAQ_NET_DAQNODE_2": f"{daq_prefix}.20",
            "DAQ_NET_GATEWAY": f"{daq_prefix}.254",
            "DAQ_NET_TESTER": f"{daq_prefix}.5",
            "QUABO_NET_MOCK": f"{quabo_prefix}.32",
            "QUABO_NET_TESTER": f"{quabo_prefix}.5",
            "DAQNODE_DIRECT_HOST": f"{daq_prefix}.10",
            "DAQNODE2_HOST": f"{daq_prefix}.20",
            "DAQNODE_DATA_HOST": f"{daq_prefix}.10",
            "DAQNODE_GATEWAY_HOST": f"{head_prefix}.254",
            "HEADNODE_HOST": f"{head_prefix}.22",
            "COMPOSE_PROJECT_NAME": f"{self.project_prefix}-{suite.name}"
        }

        # Include suite-specific env vars
        # Manually expand ${HEAD_NET_PREFIX} and ${DAQ_NET_PREFIX} if they appear in suite.env
        # because Docker Compose doesn't support nested expansion in .env files.
        for k, v in suite.env.items():
            val = v.replace("${HEAD_NET_PREFIX}", head_prefix).replace("${DAQ_NET_PREFIX}", daq_prefix).replace("${QUABO_NET_PREFIX}", quabo_prefix)
            expanded_env[k] = val

        # Include LOCAL_UID/GID in the env file for dev mode so 'exec' and 'up' are consistent
        if self.dev_mode:
            expanded_env.setdefault("LOCAL_UID", str(os.getuid()))
            expanded_env.setdefault("LOCAL_GID", str(os.getgid()))

        env_content = [f"{k}={v}" for k, v in expanded_env.items()]

        # Path for the temp env file
        env_path = CI_ROOT / f".env.{suite.name}.tmp"
        
        # Read base .env.ci content if it exists
        base_content = ""
        if ENV_CI_PATH.exists():
            base_content = ENV_CI_PATH.read_text() + "\n"
            
        # Write both base configuration and dynamic overrides
        env_path.write_text(base_content + "\n".join(env_content) + "\n")
        return env_path, expanded_env

    def _dev_overlay_flag(self, compose_file: str) -> str:
        """Returns '-f {dev_file}' if dev_mode is on and the .dev.yml overlay exists."""
        if not self.dev_mode:
            return ""
        stem = compose_file.removesuffix(".yml")
        dev_path = CI_ROOT / f"{stem}.dev.yml"
        if dev_path.exists():
            return f" -f {dev_path}"
        return ""

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
        dev_flag = self._dev_overlay_flag(compose_file)

        # Dynamic env templating
        env_file, expanded_env = self._generate_dynamic_env(suite)
        self._temp_envs[suite.name] = (env_file, expanded_env)

        # Merge suite env into process env for compose up
        full_env = os.environ.copy()
        full_env.update(expanded_env)
        full_env["COMPOSE_PROJECT_NAME"] = project_name
        if self.dev_mode:
            full_env.setdefault("LOCAL_UID", str(os.getuid()))
            full_env.setdefault("LOCAL_GID", str(os.getgid()))

        cmd = f"{self.container_tool} compose --env-file {env_file} -f {CI_ROOT}/{compose_file}{dev_flag} {profile_str} up -d{build_flag}"
        res = await self._run_cmd(cmd, env=full_env)
        if not res.ok:
            from rich.console import Console
            Console().print(f"[red]Failed to start container stack for {suite.name}[/red]")
            sys.exit(1)
            
        if suite.pre_run:
            from rich.console import Console
            Console().print(f"[dim]Running pre-run command for {suite.name}...[/dim]")
            # Use dynamic env file for pre-run too
            pre_cmd = f"{self.container_tool} compose --env-file {env_file} -f {CI_ROOT}/{compose_file} {profile_str} exec -T --user panoseti {suite.service} /bin/sh -c '{suite.pre_run}'"
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
            # For host-based suites, we don't have a compose project to down,
            # but we might have a SharedNetwork to clean up.
            if suite.name in self._temp_envs:
                _, expanded_env = self._temp_envs[suite.name]
                net_name = expanded_env.get("COMPOSE_PROJECT_NAME", f"pseti-v2-{suite.name}")
                await self._cleanup_orphaned_networks(net_name, quiet=quiet)
            return

        # Use temp env if it exists
        if suite.name in self._temp_envs:
            env_file, expanded_env = self._temp_envs[suite.name]
        else:
            env_file, expanded_env = ENV_CI_PATH, suite.env

        dev_flag = self._dev_overlay_flag(compose_file)

        # Merge suite env into process env for compose down
        full_env = os.environ.copy()
        full_env.update(expanded_env)
        full_env["COMPOSE_PROJECT_NAME"] = project_name

        # 1. Standard compose down
        cmd = f"{self.container_tool} compose --env-file {env_file} -f {CI_ROOT}/{compose_file}{dev_flag} {profile_str} down -v --remove-orphans"
        await self._run_cmd(cmd, env=full_env, quiet=quiet)

        # 2. Aggressive network cleanup for this project
        await self._cleanup_orphaned_networks(project_name, quiet=quiet)

    async def _cleanup_orphaned_networks(self, project_name: str, quiet: bool = False):
        """Find and remove all networks related to this project or pseti-v2-tc-* leftovers."""
        import docker
        client = docker.from_env()
        try:
            # 1. Aggressively kill and remove any containers matching the project or pseti-v2- prefix
            # This ensures that SharedNetwork removal doesn't fail with "active endpoints".
            container_patterns = [project_name, "pseti-v2-"]
            for container in client.containers.list(all=True):
                if any(p in container.name for p in container_patterns):
                    with contextlib.suppress(Exception):
                        container.stop(timeout=2)
                        # Use v=True to remove associated anonymous volumes
                        container.remove(force=True, v=True)

            # 2. Prune networks that match our naming patterns
            network_patterns = [project_name, "pseti-v2-tc-"]
            for network in client.networks.list():
                if any(p in network.name for p in network_patterns):
                    if not quiet:
                        from rich.console import Console
                        Console().print(f"[dim]Cleaning up network: {network.name}[/dim]")
                    try:
                        network.remove()
                    except Exception as e:
                        if not quiet:
                            from rich.console import Console
                            Console().print(f"[dim]Failed to remove {network.name}: {e}[/dim]")
        except Exception:
            pass

    async def _run_test_suite(self, suite: SuiteConfig, project_name: str, jobs: int | None, extra_args: list[str] | None) -> list[Result]:
        self._header(f"TESTING: {suite.name.upper()}")
        
        p = jobs or self.default_parallel
        args = suite.pytest_args + (extra_args or [])
        args_str = " ".join(args)

        assert suite.test_dir is not None, f"Must supply a test_dir for {suite=}"
        # normalized_pytest_dir = "${PSETI_CONTROL}/" + f"{suite.test_dir}"
        
        if suite.name in self._temp_envs:
            env_file, expanded_env = self._temp_envs[suite.name]
        else:
            # For host-based suites, we still want to generate a dynamic env 
            # to ensure non-overlapping networking if multiple suites run.
            env_file, expanded_env = self._generate_dynamic_env(suite)
            self._temp_envs[suite.name] = (env_file, expanded_env)

        lock = asyncio.Lock()
        
        if not suite.requires_docker:
            # Host-based execution (testcontainers)
            # Use sys.executable so pytest resolves to the same venv that is
            # running the orchestrator, regardless of PATH / uv activation state.
            cmd = f"{sys.executable} -m pytest {CI_ROOT / suite.test_dir} -v --color=no"
            if suite.parallel:
                cmd += f" -n {p}"
            if args_str:
                cmd += f" {args_str}"
            res = await self._stream(f"test.{suite.name}", cmd, lock, env=expanded_env)
            return [res]

        # Container-based execution (Docker Compose exec)
        # Inside the container, working_dir is /app. 
        # CI_ROOT corresponds to /app/src/ci
        container_pytest_dir = f"src/ci/{suite.test_dir}"
        pytest_cmd = f"pytest {container_pytest_dir} -v --color=no"
        if suite.parallel:
            pytest_cmd += f" -n {p}"
        if args_str:
            pytest_cmd += f" {args_str}"

        env_str = " ".join([f"-e {k}={v}" for k, v in expanded_env.items()])
        profile_str = " ".join([f"--profile {p}" for p in suite.profiles])

        compose_file = suite.compose_file
        if not compose_file and suite.environment:
            env_cfg = self.cfg.environments.get(suite.environment)
            if env_cfg:
                compose_file = env_cfg.compose_file

        dev_flag = self._dev_overlay_flag(compose_file)
        cmd = f"{self.container_tool} compose --env-file {env_file} -f {CI_ROOT}/{compose_file}{dev_flag} {profile_str} exec -T --user panoseti {env_str} {suite.service} {pytest_cmd}"
        res = await self._stream(f"test.{suite.name}", cmd, lock, env={"COMPOSE_PROJECT_NAME": project_name})
        return [res]

    async def _run_lint_suite(self, suite: SuiteConfig, project_name: str, target: str | None, extra_args: list[str] | None) -> list[Result]:
        self._header(f"LINTING: {suite.name.upper()}")

        extra_str = " ".join(extra_args or [])
        profile_str = " ".join([f"--profile {p}" for p in suite.profiles])

        compose_file = suite.compose_file
        if not compose_file and suite.environment:
            env_cfg = self.cfg.environments.get(suite.environment)
            if env_cfg:
                compose_file = env_cfg.compose_file

        lock = asyncio.Lock()
        if suite.name in self._temp_envs:
            env_file, expanded_env = self._temp_envs[suite.name]
        else:
            env_file, expanded_env = ENV_CI_PATH, suite.env

        dev_flag = self._dev_overlay_flag(compose_file)
        env_str = " ".join([f"-e {k}={v}" for k, v in expanded_env.items()])

        async def run_task(name: str, task_cmd: str):
            cmd = f"{self.container_tool} compose --env-file {env_file} -f {CI_ROOT}/{compose_file}{dev_flag} {profile_str} exec -T --user panoseti {env_str} {suite.service} {task_cmd} {extra_str}"
            print(f"lint {cmd=}")
            tag_text = f"[{name}] "
            return await self._stream(f"lint.{name}", cmd, lock, tag=tag_text, env={"COMPOSE_PROJECT_NAME": project_name})

        # Filter tasks if target is specified
        filtered_tasks = suite.tasks
        if target and target != "all":
            filtered_tasks = {n: c for n, c in suite.tasks.items() if target in n}
            if not filtered_tasks:
                 from rich.console import Console
                 Console().print(f"[yellow]No lint tasks matching '{target}' found.[/yellow]")
                 return []

        results = await asyncio.gather(*[run_task(n, c) for n, c in filtered_tasks.items()])
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
        return await stream_test_output(name, proc, lock, start, tag=tag)

    def _header(self, text: str):
        from rich.console import Console
        from rich.panel import Panel
        Console().print(Panel(f"[bold]{text}[/bold]", expand=False))

def get_isolated_env() -> dict[str, str]:
    """
    Returns a dictionary of environment variables necessary to propagate 
    the isolated test environment (PSETI_CONFIG, PSETI_STATE, etc.) 
    to subprocesses.
    """
    import os
    env = os.environ.copy()
    
    # Core isolation vars
    for var in ["PSETI_CONFIG", "PSETI_STATE", "PSETI_CONTROL", "PSETI_TMP", 
                "PSETI_LOGS", "PSETI_QUABOS", "PSETI_TQ_DIR",
                "DAQ_DATA_DIR", "HEAD_DATA_DIR", 
                "REDIS_HOST", "REDIS_PORT", "REDIS_DB",
                "LOKI_URL", "LOKI_TENANT_ID"]:
        if var in os.environ:
            env[var] = os.environ[var]
            
    # Python path to ensure src/ is importable
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = "src"
    else:
        if "src" not in env["PYTHONPATH"]:
            env["PYTHONPATH"] = f"src:{env['PYTHONPATH']}"
            
    # Ensure PSETI_ROOT points to the real repo root so src/ logic works
    from control.utils.paths import PanoPaths
    env["PSETI_ROOT"] = str(PanoPaths.software_root_dir())

    return env
