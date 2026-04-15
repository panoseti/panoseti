#!/usr/bin/env python3
"""
qa.py — PANOSETI Control Unified QA Runner

Output streams in real time. Parallel tasks (lint) prefix every line with
the task name so concurrent streams never mangle each other. Sequential
tasks (unit, integration) stream without a prefix — the section header is enough.

Usage:
  python ci/qa.py up
  python ci/qa.py down
  python ci/qa.py build
  python ci/qa.py restart
  python ci/qa.py lint
  python ci/qa.py unit [-j N] [pytest args...]
  python ci/qa.py integration [pytest args...]
  python ci/qa.py all [-j N]
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import tomllib


class C:
    """ANSI colour helpers."""

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


# Colorwheel used to assign a distinct hue to each parallel task.
PALETTE = [
    "\033[38;5;81m",   # sky blue
    "\033[38;5;118m",  # lime green
    "\033[38;5;214m",  # orange
    "\033[38;5;207m",  # pink / magenta
    "\033[38;5;147m",  # soft purple
    "\033[38;5;43m",   # teal
]


class Result:
    """Outcome of a single QA task."""

    __slots__ = ("name", "code", "elapsed", "stats")

    def __init__(self, name: str, code: int, elapsed: float, stats: dict[str, int] | None = None) -> None:
        self.name    = name
        self.code    = code
        self.elapsed = elapsed
        self.stats   = stats or {}

    @property
    def ok(self) -> bool:
        return self.code == 0


QA_TOML_PATH = Path(__file__).parent / "qa.toml"


class QARunner:
    """Loads qa.toml and drives linting / testing tasks."""

    def __init__(self, config_path: Path) -> None:
        try:
            with open(config_path, "rb") as fh:
                self._cfg: dict[str, Any] = tomllib.load(fh)
        except FileNotFoundError:
            print(C.red(f"Error: {config_path} not found."), file=sys.stderr)
            sys.exit(1)
        self._settings: dict[str, Any] = self._cfg.get("settings", {})

    @property
    def default_parallel(self) -> int:
        return int(self._settings.get("default_parallel", 4))

    def infra_task(self, kind: str) -> dict[str, str]:
        cfg: dict[str, Any] = self._cfg.get("infra", {})
        if kind not in cfg:
            return {}
        return {f"infra.{kind}": str(cfg[kind]["command"])}

    def infra_description(self, kind: str) -> str:
        cfg: dict[str, Any] = self._cfg.get("infra", {})
        return str(cfg.get(kind, {}).get("description", ""))

    def lint_tasks(self) -> dict[str, str]:
        cfg: dict[str, Any] = self._cfg.get("lint", {})
        return {f"lint.{k}": str(v["command"]) for k, v in cfg.items()}

    def lint_descriptions(self) -> dict[str, str]:
        cfg: dict[str, Any] = self._cfg.get("lint", {})
        return {f"lint.{k}": str(v.get("description", "")) for k, v in cfg.items()}

    def test_tasks(self, kind: str, parallel: int | None = None, extra_args: list[str] | None = None) -> dict[str, str]:
        cfg: dict[str, Any] = self._cfg.get("test", {})
        if kind not in cfg:
            print(C.red(f"Unknown test kind: {kind!r}"), file=sys.stderr)
            return {}
        p   = parallel if parallel is not None else self.default_parallel
        args_str = " ".join(extra_args) if extra_args else ""
        cmd = str(cfg[kind]["command"]).format(parallel=p, args=args_str)
        return {f"test.{kind}": cmd}

    def test_description(self, kind: str) -> str:
        test_cfg: dict[str, Any] = self._cfg.get("test", {})
        entry: dict[str, Any]    = test_cfg.get(kind, {})
        return str(entry.get("description", ""))

    async def check_docker(self) -> None:
        # Simple check for running containers — erroring if Docker is down.
        proc = await asyncio.create_subprocess_shell(
            "docker ps",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        if proc.returncode != 0:
            print(C.bold(C.red("⚠  Docker is not running.")))
            sys.exit(1)

    @staticmethod
    async def _stream(
        name: str,
        cmd: str,
        lock: asyncio.Lock,
        tag: str = "",
    ) -> Result:
        start = time.monotonic()
        proc  = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None

        # worker_colors maps [gw0], [gw1], etc. to distinct PALETTE entries.
        worker_colors: dict[str, str] = {}
        stats = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}

        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            upper_line = line.upper()
            
            # Identify result lines (both parallel [gwX] and sequential)
            is_result = any(kw in upper_line for kw in [" PASSED", " FAILED", " SKIPPED", " ERROR"])
            
            if "::" in line and is_result:
                if " PASSED" in upper_line: stats["passed"] += 1
                elif " FAILED" in upper_line: stats["failed"] += 1
                elif " SKIPPED" in upper_line: stats["skipped"] += 1
                elif " ERROR" in upper_line: stats["error"] += 1

            # Detect pytest-xdist worker prefixes like [gw0]
            if line.startswith("[gw"):
                end_bracket = line.find("]")
                if end_bracket != -1:
                    worker_id = line[:end_bracket + 1]
                    rest = line[end_bracket + 1:]
                    
                    if worker_id not in worker_colors:
                        worker_colors[worker_id] = PALETTE[len(worker_colors) % len(PALETTE)]
                    
                    line = f"{C.paint(worker_id, worker_colors[worker_id])}{rest}"
            
            # Suppress redundant "starting test" declarations in parallel mode.
            # These lines contain "::" but do not start with a worker ID bracket
            # AND they are not result lines.
            elif "::" in line and not line.startswith("[") and not is_result:
                continue

            async with lock:
                print(f"{tag}{line}", flush=True)

        await proc.wait()
        return Result(name, proc.returncode or 0, time.monotonic() - start, stats)

    @staticmethod
    def _header(title: str) -> None:
        bar = "─" * 60
        print(f"\n{C.bold(C.yellow(bar))}", flush=True)
        print(f"{C.bold(C.yellow(f'  {title}'))}", flush=True)
        print(f"{C.bold(C.yellow(bar))}", flush=True)

    @staticmethod
    def _task_line(name: str, desc: str, cmd: str) -> None:
        print(f"  {C.cyan(f'[{name}]')}  {desc}", flush=True)
        print(f"  {C.dim(cmd)}", flush=True)

    @staticmethod
    def _summary(
        results: list[Result],
        colors: dict[str, str] | None = None,
    ) -> None:
        if not results:
            return
        
        # 1. Individual Task Status
        width = max(len(r.name) for r in results)
        print(f"\n{C.bold('Execution Summary')}", flush=True)
        for r in results:
            icon   = C.green("✓") if r.ok else C.red("✗")
            status = C.green("passed") if r.ok else C.red("FAILED")
            code   = (colors or {}).get(r.name, C._CYAN)
            name   = C.paint(r.name.ljust(width), code)
            print(f"  {icon}  {name}  {status}  {C.dim(f'{r.elapsed:.1f}s')}", flush=True)

        # 2. Test Stats Table (if any test tasks were run)
        test_results = [r for r in results if r.name.startswith("test.") and any(r.stats.values())]
        if test_results:
            print(f"\n{C.bold('Test Metrics')}", flush=True)
            header = f"  {'Suite':<20} {'Passed':>8} {'Failed':>8} {'Skipped':>8} {'Error':>8} {'Total':>8}"
            bar = "  " + "─" * (len(header) - 2)
            print(C.dim(bar))
            print(C.bold(C.yellow(header)))
            print(C.dim(bar))

            totals = {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "total": 0}

            for r in test_results:
                s = r.stats
                passed = s.get("passed", 0)
                failed = s.get("failed", 0)
                skipped = s.get("skipped", 0)
                error = s.get("error", 0)
                total = passed + failed + skipped + error
                
                p_val = str(passed).rjust(8)
                f_val = str(failed).rjust(8)
                s_val = str(skipped).rjust(8)
                e_val = str(error).rjust(8)
                t_val = str(total).rjust(8)

                p_str = C.green(p_val) if passed > 0 else p_val
                f_str = C.red(f_val) if failed > 0 else f_val
                s_str = C.yellow(s_val) if skipped > 0 else s_val
                e_str = C.red(e_val) if error > 0 else e_val

                print(f"  {r.name:<20} {p_str} {f_str} {s_str} {e_str} {t_val}")
                
                totals["passed"] += passed
                totals["failed"] += failed
                totals["skipped"] += skipped
                totals["error"] += error
                totals["total"] += total

            print(C.dim(bar))
            p_tot = C.green(str(totals["passed"]).rjust(8)) if totals["passed"] > 0 else str(totals["passed"]).rjust(8)
            f_tot = C.red(str(totals["failed"]).rjust(8)) if totals["failed"] > 0 else str(totals["failed"]).rjust(8)
            s_tot = C.yellow(str(totals["skipped"]).rjust(8)) if totals["skipped"] > 0 else str(totals["skipped"]).rjust(8)
            e_tot = C.red(str(totals["error"]).rjust(8)) if totals["error"] > 0 else str(totals["error"]).rjust(8)
            
            print(f"  {'Total':<20} {p_tot} {f_tot} {s_tot} {e_tot} {str(totals['total']).rjust(8)}")
            print(C.dim(bar) + "\n")

    async def run_parallel(
        self,
        title: str,
        tasks: dict[str, str],
        descriptions: dict[str, str] | None = None,
    ) -> list[Result]:
        self._header(title)
        if not tasks:
            return []

        task_colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(tasks)}
        descs = descriptions or {}
        for name, cmd in tasks.items():
            colored_name = C.paint(f"[{name}]", task_colors[name])
            print(f"  {colored_name}  {descs.get(name, '')}", flush=True)
        print(flush=True)

        lock = asyncio.Lock()
        results = list(await asyncio.gather(
            *[
                self._stream(n, c, lock, tag=C.paint(f"[{n}]", task_colors[n]) + " ")
                for n, c in tasks.items()
            ]
        ))
        return results

    async def run_sequential(
        self,
        title: str,
        tasks: dict[str, str],
        descriptions: dict[str, str] | None = None,
    ) -> list[Result]:
        self._header(title)
        if not tasks:
            return []

        descs   = descriptions or {}
        lock    = asyncio.Lock()
        results: list[Result] = []

        for name, cmd in tasks.items():
            self._task_line(name, descs.get(name, ""), cmd)
            print(flush=True)
            result = await self._stream(name, cmd, lock)
            results.append(result)
            icon = C.green("✓ passed") if result.ok else C.red("✗ FAILED")
            print(f"\n{C.cyan(f'[{name}]')} {icon}  {C.dim(f'{result.elapsed:.1f}s')}", flush=True)

        return results


# ── Command handlers ───────────────────────────────────────────────────────────

async def cmd_infra(args: argparse.Namespace, runner: QARunner) -> bool:
    kind    = args.command
    tasks   = runner.infra_task(kind)
    descs   = {f"infra.{kind}": runner.infra_description(kind)}
    results = await runner.run_sequential("INFRASTRUCTURE", tasks, descs)
    runner._summary(results)
    return all(r.ok for r in results)


async def cmd_lint(args: argparse.Namespace, runner: QARunner) -> bool:
    await runner.check_docker()
    tasks   = runner.lint_tasks()
    descs   = runner.lint_descriptions()
    results = await runner.run_parallel("LINTING", tasks, descs)
    runner._summary(results)
    return all(r.ok for r in results)


async def cmd_unit(args: argparse.Namespace, runner: QARunner) -> bool:
    await runner.check_docker()
    jobs    = getattr(args, "jobs", None)
    tasks   = runner.test_tasks("unit", jobs, getattr(args, "extra", []))
    descs   = {"test.unit": runner.test_description("unit")}
    results = await runner.run_sequential("UNIT TESTS", tasks, descs)
    runner._summary(results)
    return all(r.ok for r in results)


async def cmd_integration(args: argparse.Namespace, runner: QARunner) -> bool:
    await runner.check_docker()
    tasks   = runner.test_tasks("integration", extra_args=getattr(args, "extra", []))
    descs   = {"test.integration": runner.test_description("integration")}
    results = await runner.run_sequential("INTEGRATION TESTS", tasks, descs)
    runner._summary(results)
    return all(r.ok for r in results)


async def cmd_all(args: argparse.Namespace, runner: QARunner) -> bool:
    await runner.check_docker()
    all_results: list[Result] = []
    
    all_results += await runner.run_parallel(
        "LINTING", runner.lint_tasks(), runner.lint_descriptions(),
    )
    
    all_results += await runner.run_sequential(
        "UNIT TESTS", runner.test_tasks("unit", getattr(args, "jobs", None)),
        {"test.unit": runner.test_description("unit")}
    )
    
    all_results += await runner.run_sequential(
        "INTEGRATION TESTS", runner.test_tasks("integration"),
        {"test.integration": runner.test_description("integration")}
    )

    width = 60
    print(f"\n{C.bold(C.cyan('═' * width))}")
    print(f"{C.bold(C.cyan('  Summary'))}")
    print(f"{C.bold(C.cyan('═' * width))}")
    runner._summary(all_results)
    return all(r.ok for r in all_results)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python ci/qa.py")
    sub    = parser.add_subparsers(dest="command")

    # Infra commands
    for infra in ["up", "down", "build", "restart"]:
        sub.add_parser(infra, help=f"Infra: {infra}")
    
    sub.add_parser("lint", help="Run linters")
    
    p_unit = sub.add_parser("unit", help="Run unit tests")
    p_unit.add_argument("-j", "--jobs", type=int, default=None, help="Parallel workers")
    p_unit.add_argument("extra", nargs="*", help="Extra pytest arguments")

    p_int = sub.add_parser("integration", help="Run integration tests")
    p_int.add_argument("extra", nargs="*", help="Extra pytest arguments")
    
    p_all = sub.add_parser("all", help="Run full suite")
    p_all.add_argument("-j", "--jobs", type=int, default=None, help="Parallel workers for unit tests")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    runner = QARunner(QA_TOML_PATH)

    try:
        # Determine if it's an infra command or a QA command
        if args.command in ["up", "down", "build", "restart"]:
            ok = asyncio.run(cmd_infra(args, runner))
        else:
            ok = asyncio.run(getattr(sys.modules[__name__], f"cmd_{args.command}")(args, runner))
    except KeyboardInterrupt:
        sys.exit(130)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
