#!/usr/bin/env python3
"""
qa.py — PANOSETI Control Unified QA Runner (Modernized)

This runner provides modular, isolated, and strongly-typed test execution.
It uses Pydantic to validate test suite configurations defined in qa.toml.
"""

import asyncio
import json
import os
import re
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, Literal

import typer
from pydantic import BaseModel, Field, field_validator

# ── ANSI Colors ──────────────────────────────────────────────────────────────

class C:
    _GREEN  = "\033[92m"
    _RED    = "\033[91m"
    _YELLOW = "\033[93m"
    _CYAN   = "\033[96m"
    _BOLD   = "\033[1m"
    _DIM    = "\033[2m"
    _RESET  = "\033[0m"

    @staticmethod
    def green(s: str)  -> str: return f"{C._GREEN}{s}{C._RESET}"
    @staticmethod
    def red(s: str)    -> str: return f"{C._RED}{s}{C._RESET}"
    @staticmethod
    def yellow(s: str) -> str: return f"{C._YELLOW}{s}{C._RESET}"
    @staticmethod
    def cyan(s: str)   -> str: return f"{C._CYAN}{s}{C._RESET}"
    @staticmethod
    def bold(s: str)   -> str: return f"{C._BOLD}{s}{C._RESET}"
    @staticmethod
    def dim(s: str)    -> str: return f"{C._DIM}{s}{C._RESET}"
    @staticmethod
    def paint(s: str, code: str) -> str: return f"{code}{s}{C._RESET}"

PALETTE = [
    "\033[38;5;81m",   # sky blue
    "\033[38;5;118m",  # lime green
    "\033[38;5;214m",  # orange
    "\033[38;5;207m",  # pink / magenta
    "\033[38;5;147m",  # soft purple
    "\033[38;5;43m",   # teal
]

# ── Configuration Models ──────────────────────────────────────────────────────

class SuiteConfig(BaseModel):
    name: str = ""
    description: str = ""
    type: Literal["test", "lint"] = "test"
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

class QAConfig(BaseModel):
    settings: dict[str, Any] = Field(default_factory=dict)
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
    __slots__ = ("code", "elapsed", "name", "stats")
    def __init__(self, name: str, code: int, elapsed: float, stats: dict[str, int] | None = None) -> None:
        self.name    = name
        self.code    = code
        self.elapsed = elapsed
        self.stats   = stats or {}
    @property
    def ok(self) -> bool: return self.code == 0

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
            print(C.red(f"Error loading {config_path}: {e}"), file=sys.stderr)
            sys.exit(1)
        
        self.no_teardown = False
        self.default_parallel = self.cfg.settings.get("default_parallel", 4)
        self.project_prefix = self.cfg.settings.get("project_prefix", "pseti")

    async def run_suite(self, suite_name: str, jobs: int | None = None, extra_args: list[str] | None = None) -> bool:
        if suite_name not in self.cfg.suites:
            print(C.red(f"Unknown suite: {suite_name}"), file=sys.stderr)
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
        for name, suite in self.cfg.suites.items():
            if suite.requires_docker:
                project_name = f"{self.project_prefix}-{suite.name}"
                print(C.dim(f"Cleaning up {project_name}..."))
                await self._teardown_docker(suite, project_name, quiet=True)
        print(C.green("Cleanup complete."))

    async def build_all(self):
        self._header("REBUILDING IMAGES")
        processed_files = set()
        for suite in self.cfg.suites.values():
            if suite.requires_docker and suite.compose_file not in processed_files:
                project_name = f"{self.project_prefix}-build"
                cmd = f"docker compose --env-file {ENV_CI_PATH} -f {CONTROL_ROOT}/{suite.compose_file} build"
                await self._run_cmd(cmd, env={"COMPOSE_PROJECT_NAME": project_name})
                processed_files.add(suite.compose_file)

    # ── Internal Helpers ──────────────────────────────────────────────────────

    async def _setup_docker(self, suite: SuiteConfig, project_name: str):
        self._header(f"SETUP: {suite.name.upper()}")
        profile_str = " ".join([f"--profile {p}" for p in suite.profiles])
        cmd = f"docker compose --env-file {ENV_CI_PATH} -f {CONTROL_ROOT}/{suite.compose_file} {profile_str} up -d"
        res = await self._run_cmd(cmd, env={"COMPOSE_PROJECT_NAME": project_name})
        if not res.ok:
            print(C.red(f"Failed to start Docker stack for {suite.name}"), file=sys.stderr)
            sys.exit(1)
            
        if suite.pre_run:
            print(C.dim(f"Running pre-run command for {suite.name}..."))
            pre_cmd = f"docker compose --env-file {ENV_CI_PATH} -f {CONTROL_ROOT}/{suite.compose_file} {profile_str} exec -T {suite.service} /bin/sh -c '{suite.pre_run}'"
            await self._run_cmd(pre_cmd, env={"COMPOSE_PROJECT_NAME": project_name})

    async def _teardown_docker(self, suite: SuiteConfig, project_name: str, quiet: bool = False):
        if not quiet:
            self._header(f"TEARDOWN: {suite.name.upper()}")
        profile_str = " ".join([f"--profile {p}" for p in suite.profiles])
        cmd = f"docker compose --env-file {ENV_CI_PATH} -f {CONTROL_ROOT}/{suite.compose_file} {profile_str} down -v --remove-orphans"
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
        
        cmd = f"docker compose --env-file {ENV_CI_PATH} -f {CONTROL_ROOT}/{suite.compose_file} {profile_str} exec -T {env_str} {suite.service} {pytest_cmd}"
        
        lock = asyncio.Lock()
        res = await self._stream(f"test.{suite.name}", cmd, lock, env={"COMPOSE_PROJECT_NAME": project_name})
        return [res]

    async def _run_lint_suite(self, suite: SuiteConfig, project_name: str, extra_args: list[str] | None) -> list[Result]:
        self._header(f"LINTING: {suite.name.upper()}")
        
        extra_str = " ".join(extra_args or [])
        task_colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(suite.tasks)}
        profile_str = " ".join([f"--profile {p}" for p in suite.profiles])
        
        lock = asyncio.Lock()
        
        async def run_task(name: str, task_cmd: str):
            cmd = f"docker compose --env-file {ENV_CI_PATH} -f {CONTROL_ROOT}/{suite.compose_file} {profile_str} exec -T {suite.service} {task_cmd} {extra_str}"
            return await self._stream(f"lint.{name}", cmd, lock, tag=C.paint(f"[{name}]", task_colors[name]) + " ", env={"COMPOSE_PROJECT_NAME": project_name})

        results = await asyncio.gather(*[run_task(n, c) for n, c in suite.tasks.items()])
        return list(results)

    async def _run_cmd(self, cmd: str, env: dict[str, str] | None = None, quiet: bool = False) -> Result:
        start = time.monotonic()
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
            
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.DEVNULL if quiet else None,
            stderr=asyncio.subprocess.DEVNULL if quiet else None,
            env=full_env
        )
        await proc.wait()
        return Result("cmd", proc.returncode or 0, time.monotonic() - start)

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

        worker_colors: dict[str, str] = {}
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
                except: pass

            upper_line = plain_line.upper()
            if not has_json_metrics:
                is_result = (plain_line.startswith("[gw") and any(kw in upper_line for kw in [" PASSED", " FAILED", " SKIPPED", " ERROR", " XFAIL", " XPASS"])) or \
                            ("::" in plain_line and any(upper_line.endswith(kw) or f"{kw} [" in upper_line for kw in ["PASSED", "FAILED", "SKIPPED", "ERROR", "XFAIL", "XPASS"]))
                if is_result:
                    if "PASSED" in upper_line or "XPASS" in upper_line: stats["passed"] += 1
                    elif "FAILED" in upper_line: stats["failed"] += 1
                    elif "SKIPPED" in upper_line or "XFAIL" in upper_line: stats["skipped"] += 1
                    elif "ERROR" in upper_line: stats["error"] += 1

            if line.startswith("[gw"):
                end = line.find("]")
                if end != -1:
                    wid = line[:end+1]
                    if wid not in worker_colors: worker_colors[wid] = PALETTE[len(worker_colors) % len(PALETTE)]
                    line = f"{C.paint(wid, worker_colors[wid])}{line[end+1:]}"
            elif is_parallel and "::" in plain_line and not plain_line.startswith("["):
                if not any(kw in upper_line for kw in ["PASSED", "FAILED", "SKIPPED", "ERROR", "XFAIL", "XPASS"]): continue

            async with lock:
                print(f"{tag}{line}", flush=True)

        await proc.wait()
        return Result(name, proc.returncode or 0, time.monotonic() - start, stats)

    def _header(self, title: str):
        bar = "─" * 60
        print(f"\n{C.bold(C.yellow(bar))}\n{C.bold(C.yellow(f'  {title}'))}\n{C.bold(C.yellow(bar))}", flush=True)

    def _print_summary(self, results: list[Result]):
        if not results: return
        width = max(len(r.name) for r in results)
        print(f"\n{C.bold('Execution Summary')}")
        for r in results:
            icon = C.green("✓") if r.ok else C.red("✗")
            status = C.green("passed") if r.ok else C.red("FAILED")
            print(f"  {icon}  {C.cyan(r.name.ljust(width))}  {status}  {C.dim(f'{r.elapsed:.1f}s')}")
        
        test_res = [r for r in results if r.name.startswith("test.") and any(r.stats.values())]
        if test_res:
            self._print_metrics(test_res)

    def _print_metrics(self, test_results: list[Result]):
        print(f"\n{C.bold('Test Metrics')}")
        header = f"  {'Suite':<20} {'Passed':>8} {'Failed':>8} {'Skipped':>8} {'Error':>8} {'Total':>8}"
        bar = "  " + "─" * (len(header) - 2)
        print(C.dim(bar))
        print(C.bold(C.yellow(header)))
        print(C.dim(bar))
        totals = {"p": 0, "f": 0, "s": 0, "e": 0, "t": 0}
        for r in test_results:
            s = r.stats
            p, f, sk, e = s.get("passed", 0), s.get("failed", 0), s.get("skipped", 0), s.get("error", 0)
            t = p + f + sk + e
            print(f"  {r.name:<20} {C.green(str(p).rjust(8)) if p else str(p).rjust(8)} {C.red(str(f).rjust(8)) if f else str(f).rjust(8)} {C.yellow(str(sk).rjust(8)) if sk else str(sk).rjust(8)} {C.red(str(e).rjust(8)) if e else str(e).rjust(8)} {str(t).rjust(8)}")
            totals["p"] += p; totals["f"] += f; totals["s"] += sk; totals["e"] += e; totals["t"] += t
        print(C.dim(bar))
        print(f"  {'Total':<20} {C.green(str(totals['p']).rjust(8)) if totals['p'] else str(totals['p']).rjust(8)} {C.red(str(totals['f']).rjust(8)) if totals['f'] else str(totals['f']).rjust(8)} {C.yellow(str(totals['s']).rjust(8)) if totals['s'] else str(totals['s']).rjust(8)} {C.red(str(totals['e']).rjust(8)) if totals['e'] else str(totals['e']).rjust(8)} {str(totals['t']).rjust(8)}\n")

# ── Typer CLI ─────────────────────────────────────────────────────────────────

app = typer.Typer(
    help="PANOSETI Quality Assurance & Testing Suite",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

@app.callback()
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", "--no-teardown", help="Bypass Docker teardown for debugging.")
):
    ctx.obj = TestRunner(QA_TOML_PATH)
    ctx.obj.no_teardown = debug

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def lint(ctx: typer.Context):
    """Run linters [ruff/mypy args...]"""
    ok = asyncio.run(ctx.obj.run_suite("lint", extra_args=ctx.args))
    if not ok: raise typer.Exit(code=1)

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def unit(ctx: typer.Context, jobs: int | None = typer.Option(None, "--jobs", "-j")):
    """Run unit tests [-j N] [pytest args...]"""
    ok = asyncio.run(ctx.obj.run_suite("unit", jobs=jobs, extra_args=ctx.args))
    if not ok: raise typer.Exit(code=1)

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def integration(ctx: typer.Context):
    """Run integration tests [pytest args...]"""
    ok = asyncio.run(ctx.obj.run_suite("integration", extra_args=ctx.args))
    if not ok: raise typer.Exit(code=1)

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def chaos(ctx: typer.Context):
    """Run chaos scenario tests [pytest args...]"""
    ok = asyncio.run(ctx.obj.run_suite("chaos", extra_args=ctx.args))
    if not ok: raise typer.Exit(code=1)

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def structural(ctx: typer.Context):
    """Run structural tests [pytest args...]"""
    ok = asyncio.run(ctx.obj.run_suite("structural", extra_args=ctx.args))
    if not ok: raise typer.Exit(code=1)

@app.command()
def build():
    """Rebuild all test images"""
    runner = TestRunner(QA_TOML_PATH)
    asyncio.run(runner.build_all())

@app.command()
def cleanup():
    """Tear down all test containers and volumes"""
    runner = TestRunner(QA_TOML_PATH)
    asyncio.run(runner.cleanup_all())

@app.command(name="all", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def all_tests(ctx: typer.Context, jobs: int | None = typer.Option(None, "--jobs", "-j")):
    """Run the full testing suite"""
    suites = ["lint", "unit", "structural", "integration", "chaos"]
    success = True
    for s in suites:
        ok = asyncio.run(ctx.obj.run_suite(s, jobs=jobs, extra_args=ctx.args))
        success = success and ok
    if not success: raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
