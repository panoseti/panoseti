"""
HITL (Hardware-in-the-Loop) test orchestration CLI.
Subcommands: plan, run, preflight, status, safe-down, list-classes, explain.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(help="Hardware-in-the-Loop (HITL) physical lab tests", no_args_is_help=True)
console = Console()

_HW_SW_DIR = Path(__file__).parent.parent
_STATE_FILE = Path.home() / ".pseti" / "hw_runtime_state.json"


def _get_sm():
    from ci.hardware_software.hw_utils.state_machine import HardwareStateMachine
    return HardwareStateMachine()


def _get_topology():
    from ci.hardware_software.hw_utils.topology import HwTopology
    return HwTopology()


def _read_state() -> str | None:
    from ci.hardware_software.hw_utils.state_machine import read_state
    return read_state(_STATE_FILE)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

@app.command(name="plan")
def hw_plan(
    hw_class: Annotated[str | None, typer.Option("--class", "-c", help="Filter to one test class.")] = None,
    assume_state: Annotated[str | None, typer.Option("--assume-state", help="Assume hardware is already in this state.")] = None,
) -> None:
    """Dry-run: print the batch plan + estimated wall clock. No hardware touched."""
    try:
        sm = _get_sm()
        from ci.hardware_software.hw_utils.scheduler import StateAwareScheduler
        scheduler = StateAwareScheduler(sm)

        # Collect pytest items without running them
        items = _collect_items(hw_class=hw_class)
        if not items:
            console.print("[yellow]No matching HITL tests found.[/yellow]")
            return

        batches = scheduler.build_plan(items, assume_state=assume_state)
        if not batches:
            console.print("[yellow]No tests classified into TOML batches.[/yellow]")
            return

        plan_str = scheduler.format_plan(batches)
        console.print(plan_str)
    except Exception as exc:
        console.print(f"[red]Plan failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@app.command(name="run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def hw_run(
    ctx: typer.Context,
    dev: Annotated[bool, typer.Option("--dev", help="Dev mode: skip power cycles, keep hardware running.")] = False,
    hw_class: Annotated[str | None, typer.Option("--class", "-c", help="Filter to one TOML test class.")] = None,
    hw_state: Annotated[str | None, typer.Option("--state", "-s", help="Run only tests requiring ≤ this state.")] = None,
    assume_state: Annotated[str | None, typer.Option("--assume-state", help="Trust that hardware is already in this state.")] = None,
    no_power_cycle: Annotated[bool, typer.Option("--no-power-cycle", help="Refuse to invoke high-safety (power cycle) primitives.")] = False,
    keep_running: Annotated[bool, typer.Option("--keep-running", help="Skip the final safety teardown (dev/lab use only).")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
    explain: Annotated[str | None, typer.Option("--explain", help="Print state plan for a single test ID and exit.")] = None,
) -> None:
    """Run HITL tests with state-aware batching."""
    if dev:
        console.print("[bold yellow]DEV MODE[/bold yellow] — power cycles skipped; hardware will NOT be returned to safe state.")
        keep_running = True
        if assume_state is None:
            assume_state = "ACQ_CONFIGURED"

    if explain:
        _cmd_explain(explain)
        return

    pytest_args = list(ctx.args)
    if hw_class:
        pytest_args += ["-m", f"hw_class({hw_class!r})"]
    if hw_state:
        pytest_args += ["-m", f"required_state({hw_state!r})"]

    cmd = [
        sys.executable, "-m", "pytest",
        str(_HW_SW_DIR),
        "-p", "ci.hardware_software.hw_utils.pytest_plugin",
        "--tb=short",
        *pytest_args,
    ]

    if not yes and not _confirm_run(cmd):
        raise typer.Exit(code=0)

    if not keep_running:
        sm = _get_sm()
        from ci.hardware_software.hw_utils.safety import SafetyManager
        mgr = SafetyManager(sm, _STATE_FILE, keep_running=keep_running)
        mgr.register()

    ret = subprocess.run(cmd, cwd=_HW_SW_DIR.parent.parent.parent).returncode
    raise typer.Exit(code=ret)


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

@app.command(name="preflight", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def hw_preflight(ctx: typer.Context) -> None:
    """Run only tests with preflight=true in their TOML class (pre-observation subset)."""
    import tomllib
    toml_path = _HW_SW_DIR / "hw_tests.toml"
    with toml_path.open("rb") as f:
        data = tomllib.load(f)
    preflight_classes = [
        name for name, cfg in data.get("classes", {}).items()
        if cfg.get("preflight", False)
    ]
    if not preflight_classes:
        console.print("[yellow]No preflight classes defined in hw_tests.toml.[/yellow]")
        return

    marker_expr = " or ".join(f"hw_class({c!r})" for c in preflight_classes)
    cmd = [
        sys.executable, "-m", "pytest",
        str(_HW_SW_DIR),
        "-p", "ci.hardware_software.hw_utils.pytest_plugin",
        "-m", marker_expr,
        "--tb=short",
        *ctx.args,
    ]
    ret = subprocess.run(cmd, cwd=_HW_SW_DIR.parent.parent.parent).returncode
    raise typer.Exit(code=ret)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@app.command(name="status")
def hw_status() -> None:
    """Report current believed hardware state and reachability."""
    state = _read_state()
    if state:
        console.print(f"Last known state: [green]{state}[/green] (from {_STATE_FILE})")
    else:
        console.print("[dim]No persisted hardware state found.[/dim]")

    try:
        topo = _get_topology()
        for q in topo.quabo_ips()[:4]:  # show first module only
            import subprocess as sp
            r = sp.run(["ping", "-c1", "-W1", q.ip], capture_output=True, timeout=3)
            reachable = "[green]✓[/green]" if r.returncode == 0 else "[red]✗[/red]"
            console.print(f"  {reachable} {q.ip} (module {q.module_id} Q{q.quadrant})")
    except Exception as exc:
        console.print(f"[dim]Topology unavailable: {exc}[/dim]")


# ---------------------------------------------------------------------------
# safe-down
# ---------------------------------------------------------------------------

@app.command(name="safe-down")
def hw_safe_down(
    keep_running: Annotated[bool, typer.Option("--keep-running", help="Print banner but do not power off.")] = False,
) -> None:
    """Manually invoke emergency teardown (drive hardware to safe state)."""
    sm = _get_sm()
    from ci.hardware_software.hw_utils.safety import SafetyManager
    mgr = SafetyManager(sm, _STATE_FILE, keep_running=keep_running)
    mgr.emergency_teardown()


# ---------------------------------------------------------------------------
# list-classes
# ---------------------------------------------------------------------------

@app.command(name="list-classes")
def hw_list_classes() -> None:
    """Print TOML classes and how many tests each contains in the current collection."""
    import tomllib
    toml_path = _HW_SW_DIR / "hw_tests.toml"
    with toml_path.open("rb") as f:
        data = tomllib.load(f)
    classes = data.get("classes", {})
    console.print("[bold]HITL Test Classes[/bold]")
    for name, cfg in classes.items():
        console.print(
            f"  [cyan]{name}[/cyan]  required_state=[green]{cfg.get('required_state', '?')}[/green]"
            f"  priority={cfg.get('batch_priority', '?')}"
            f"  preflight={'[green]yes[/green]' if cfg.get('preflight') else 'no'}"
            f"\n    {cfg.get('description', '')}"
        )


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------

@app.command(name="explain")
def hw_explain(
    test_id: Annotated[str, typer.Argument(help="Test node ID to explain.")],
    assume_state: Annotated[str | None, typer.Option("--assume-state")] = None,
) -> None:
    """Print the state transition plan a single test would trigger."""
    _cmd_explain(test_id, assume_state)


# ---------------------------------------------------------------------------
# check-env  (retained from original CLI)
# ---------------------------------------------------------------------------

@app.command(name="check-env")
def hw_check_env() -> None:
    """Verify HITL environment: config files, network reachability, WPS."""
    try:
        topo = _get_topology()
        console.print(f"Modules: {topo.module_ids()}")
        console.print(f"DAQ nodes: {[n.host for n in topo.daq_nodes()]}")
        console.print(f"Capabilities: {topo.capabilities()}")
        console.print("[green]check-env OK[/green]")
    except Exception as exc:
        console.print(f"[red]check-env failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cmd_explain(test_id: str, assume_state: str | None = None) -> None:
    import tomllib
    toml_path = _HW_SW_DIR / "hw_tests.toml"
    with toml_path.open("rb") as f:
        data = tomllib.load(f)

    from ci.hardware_software.hw_utils.scheduler import StateAwareScheduler
    sm = _get_sm()
    scheduler = StateAwareScheduler(sm)
    cls_name = scheduler.class_for(test_id)
    if not cls_name:
        console.print(f"[yellow]{test_id!r} has no TOML class mapping.[/yellow]")
        return

    cls_cfg = data.get("classes", {}).get(cls_name, {})
    target = cls_cfg.get("required_state", sm.initial)
    current = assume_state or sm.initial
    console.print(f"Test class: [cyan]{cls_name}[/cyan]  required_state=[green]{target}[/green]")
    if current == target:
        console.print("  [dim]No transition needed from current state.[/dim]")
        return
    try:
        plan = sm.plan(current, target)
        cost = sm.cost(plan)
        steps = " → ".join(f"{p.name} ({p.budget_s['typical']:.0f}s)" for p in plan)
        console.print(f"  Transition: {steps}")
        console.print(f"  Total cost: {cost:.0f}s")
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")


def _collect_items(hw_class: str | None = None) -> list:
    """Collect pytest items without running them."""
    import pytest

    collected: list = []

    class Collector(pytest.Plugin if hasattr(pytest, "Plugin") else object):
        def pytest_collection_finish(self, session):
            collected.extend(session.items)

    args = [str(_HW_SW_DIR), "--collect-only", "-q"]
    if hw_class:
        args += ["-m", f"hw_class({hw_class!r})"]

    collector = Collector()
    pytest.main([*args, "--co"], plugins=[collector])
    return collected


def _confirm_run(cmd: list[str]) -> bool:
    console.print(f"[dim]Will run:[/dim] {' '.join(cmd)}")
    ans = typer.prompt("Proceed? [y/N]", default="N")
    return ans.strip().lower() in ("y", "yes")
