"""
Unified test orchestration CLI for PSETI.
Provides subcommands for Linting, Software (Docker CI), and Hardware (HITL) tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from pathlib import Path
from typing import Annotated

import click
import typer
import typer.core
from panoseti_grpc.util.cli import BaseLazyGroup, display_tree_callback
from rich.console import Console

from ci.software_only.qa_utils import (
    QA_TOML_PATH,
    TestRunner,
)
from control.utils.paths import PanoPaths

V2_QA_TOML_PATH = PanoPaths.base_dir() / "src" / "ci" / "software_only_v2" / "qa.toml"

console = Console()


class GrpcTestLazyGroup(BaseLazyGroup):
    """
    Lazy-loading group for gRPC service layer tests.
    Unwraps the tests.qa app from the grpc/ directory.
    """
    def list_commands(self, ctx: click.Context) -> list[str]:
        import importlib
        root = PanoPaths.software_root_dir()
        grpc_tests = str(root / "grpc")
        if Path(grpc_tests).exists() and grpc_tests not in sys.path:
            sys.path.insert(0, grpc_tests)
        try:
            mod = importlib.import_module("tests.qa")
            test_app = mod.app
            click_group = typer.main.get_command(test_app)
            return click_group.list_commands(ctx)  # type: ignore[attr-defined]
        except Exception:
            return []

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        import importlib
        root = PanoPaths.software_root_dir()
        grpc_tests = str(root / "grpc")
        if Path(grpc_tests).exists() and grpc_tests not in sys.path:
            sys.path.insert(0, grpc_tests)
            
        try:
            mod = importlib.import_module("tests.qa")
            test_app = mod.app
            click_group = typer.main.get_command(test_app)
            cmd = click_group.get_command(ctx, cmd_name)  # type: ignore[attr-defined]
            if cmd:
                cmd.name = cmd_name
                return cmd
            return None
        except Exception as e:
            error_console = Console(stderr=True)
            error_console.print(f"[red]Error loading gRPC test command '{cmd_name}': {e}[/red]")
            return None


class HwTestLazyGroup(BaseLazyGroup):
    """
    Lazy-loading group for hardware_software (HITL) tests.
    Delegates to ci.hardware_software.hw_utils.cli.
    """
    def list_commands(self, ctx: click.Context) -> list[str]:
        import importlib
        try:
            mod = importlib.import_module("ci.hardware_software.hw_utils.cli")
            test_app = mod.app
            click_group = typer.main.get_command(test_app)
            return click_group.list_commands(ctx)  # type: ignore[attr-defined]
        except Exception:
            return []

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        import importlib
        try:
            mod = importlib.import_module("ci.hardware_software.hw_utils.cli")
            test_app = mod.app
            click_group = typer.main.get_command(test_app)
            cmd = click_group.get_command(ctx, cmd_name)  # type: ignore[attr-defined]
            if cmd:
                cmd.name = cmd_name
                return cmd
            return None
        except Exception as e:
            error_console = Console(stderr=True)
            error_console.print(f"[red]Error loading HW test command '{cmd_name}': {e}[/red]")
            return None


app = typer.Typer(
    help="PSETI Quality Assurance & Testing Suite.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Sub-apps for organization
sw_app = typer.Typer(help="Software QA tests (Docker-based CI simulations)", no_args_is_help=True)
sw2_app = typer.Typer(help="v2 Software QA — topology-driven, realistic containers", no_args_is_help=True)
hw_app = typer.Typer(help="Hardware-in-the-Loop (HITL) physical lab tests", no_args_is_help=True, cls=HwTestLazyGroup)
grpc_app = typer.Typer(help="gRPC service layer tests", no_args_is_help=True, cls=GrpcTestLazyGroup)
v2_app = typer.Typer(help="v2 Software QA (topology-driven, realistic containers)", no_args_is_help=True)

app.add_typer(sw_app, name="sw")
app.add_typer(sw2_app, name="sw2")
app.add_typer(hw_app, name="hw")
app.add_typer(grpc_app, name="grpc")
sw_app.add_typer(v2_app, name="v2")

@grpc_app.callback()
def grpc_main(
    ctx: typer.Context,
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree for gRPC tests.", callback=display_tree_callback)] = False
) -> None:
    """gRPC service layer tests"""
    if tree:
        return
    # Ensure we are running from grpc/tests so that relative paths in qa.toml resolve
    root = PanoPaths.software_root_dir()
    grpc_tests = root / "grpc" / "tests"
    if grpc_tests.exists():
        os.chdir(grpc_tests)
        if str(grpc_tests) not in sys.path:
            sys.path.insert(0, str(grpc_tests))
        
        # Override TestRunner for gRPC context
        from grpc_qa_utils import QA_TOML_PATH as GRPC_QA_TOML
        from grpc_qa_utils import TestRunner as GrpcTestRunner
        old_runner = ctx.obj
        ctx.obj = GrpcTestRunner(GRPC_QA_TOML)
        if old_runner:
            ctx.obj.no_teardown = old_runner.no_teardown
            ctx.obj.no_build = old_runner.no_build
            ctx.obj.container_tool = old_runner.container_tool

# ---------------------------------------------------------------------------
# Global Setup
# ---------------------------------------------------------------------------

@app.command()
def prune(
    ctx: typer.Context,
    all: Annotated[bool, typer.Option("--all", "-a", help="Prune all PSETI containers and networks, including 'outer' test-runners.")] = False,
) -> None:
    """Aggressively prune PSETI test containers and networks."""
    import docker
    client = docker.from_env()

    # Base patterns for v2 testcontainers and shared networks
    container_patterns = ["pseti-v2-"]
    network_patterns = ["pseti-v2-tc-", "pseti-v2-shared-net"]

    if all:
        # Include 'outer' stack patterns used by pseti test sw/lint
        container_patterns.extend(["ctl-int-", "ci-", "pseti-lint", "pseti-v2-integration"])
        network_patterns.extend(["pseti-lint", "ctl-int", "ci-", "pseti-v2-integration"])

    console.print(f"[bold cyan]Pruning PSETI containers... (all={all})[/bold cyan]")
    containers = client.containers.list(all=True)
    for container in containers:
        if container is None or container.name is None:
            continue
        if any(p in container.name for p in container_patterns):
            console.print(f"  - Stopping/Removing [bold]{container.name}[/bold]")
            with contextlib.suppress(Exception):
                container.stop(timeout=2)
                container.remove(force=True, v=True)

    console.print("\n[bold cyan]Pruning PSETI networks...[/bold cyan]")
    for network in client.networks.list():
        if network is None or network.name is None:
            continue
        if any(p in network.name for p in network_patterns):
            console.print(f"  - Removing [bold]{network.name}[/bold]")
            with contextlib.suppress(Exception):
                network.remove()

    console.print("\n[bold cyan]Pruning PSETI volumes...[/bold cyan]")
    volume_patterns = ["daq_data", "mock_quabo_uds", "grafana_data", "ci_panoseti_mnt"]
    for volume in client.volumes.list():
        if any(p in volume.name for p in volume_patterns):
            console.print(f"  - Removing [bold]{volume.name}[/bold]")
            with contextlib.suppress(Exception):
                volume.remove(force=True)

    console.print("\n[bold green]Cleanup complete.[/bold green]")


@app.callback()
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", "--no-teardown", help="Bypass container teardown for debugging."),
    no_build: bool = typer.Option(False, "--no-build", help="Do not attempt to build images, use existing ones."),
    tool: str = typer.Option("docker", "--tool", help="Container tool to use (docker or podman)."),
    dev: bool = typer.Option(False, "--dev", help="Add .dev.yml overlay: hot-mounted source + LOCAL_UID/LOCAL_GID for UID rewrite."),
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree for PSETI testing.", callback=display_tree_callback)] = False
) -> None:
    """
    PSETI Testing Suite.
    """
    ctx.obj = TestRunner(QA_TOML_PATH)
    ctx.obj.no_teardown = debug
    ctx.obj.no_build = no_build
    ctx.obj.container_tool = tool
    ctx.obj.dev_mode = dev

@sw_app.callback()
def sw_main(
    ctx: typer.Context,
    tool: str | None = typer.Option(None, "--tool", help="Container tool to use (docker or podman)."),
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree for software tests.", callback=display_tree_callback)] = False
) -> None:
    """Software QA tests (Docker-based CI simulations)"""
    if tool and hasattr(ctx, "obj") and ctx.obj:
        ctx.obj.container_tool = tool

@hw_app.callback()
def hw_main(
    ctx: typer.Context,
    tool: str | None = typer.Option(None, "--tool", help="Container tool to use (docker or podman)."),
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree for hardware tests.", callback=display_tree_callback)] = False
) -> None:
    """Hardware-in-the-Loop (HITL) physical lab tests"""
    if tool and hasattr(ctx, "obj") and ctx.obj:
        ctx.obj.container_tool = tool

# ---------------------------------------------------------------------------
# LINT Subcommands
# ---------------------------------------------------------------------------

@app.command(name="lint", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def lint_main(
    ctx: typer.Context,
    targets: Annotated[str, typer.Argument(help="Scope to lint: 'ruff', 'mypy', or 'all'")] = "all",
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree for linting.", callback=display_tree_callback)] = False
) -> None:
    """Static analysis and linting (Ruff, MyPy)"""
    if tree:
        return
    ok = asyncio.run(ctx.obj.run_suite("lint", target=targets, extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# SW Subcommands (Software QA)
# ---------------------------------------------------------------------------

@sw_app.command(name="unit", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sw_unit(
    ctx: typer.Context, 
    jobs: int | None = typer.Option(None, "--jobs", "-j", help="Parallel jobs"),
    tool: str | None = typer.Option(None, "--tool", help="Container tool to use (docker or podman).")
) -> None:
    """Tier 1: Parallel unit tests (Logic & Parsing)"""
    if tool and hasattr(ctx, "obj") and ctx.obj:
        ctx.obj.container_tool = tool
    ok = asyncio.run(ctx.obj.run_suite("unit", jobs=jobs, extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

@sw_app.command(name="logic", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sw_logic(
    ctx: typer.Context,
    tool: str | None = typer.Option(None, "--tool", help="Container tool to use (docker or podman).")
) -> None:
    """Tier 2: Subsystem logic (Isolated State, Mocked gRPC)"""
    if tool and hasattr(ctx, "obj") and ctx.obj:
        ctx.obj.container_tool = tool
    ok = asyncio.run(ctx.obj.run_suite("logic", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

@sw_app.command(name="fleet", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sw_fleet(
    ctx: typer.Context,
    tool: str | None = typer.Option(None, "--tool", help="Container tool to use (docker or podman).")
) -> None:
    """Tier 3: Distributed flows (Dynamic Fleets, Real gRPC)"""
    if tool and hasattr(ctx, "obj") and ctx.obj:
        ctx.obj.container_tool = tool
    ok = asyncio.run(ctx.obj.run_suite("fleet", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

@sw_app.command(name="chaos", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sw_chaos(
    ctx: typer.Context,
    tool: str | None = typer.Option(None, "--tool", help="Container tool to use (docker or podman).")
) -> None:
    """Tier 4: Fault injection (Active failure scenarios)"""
    if tool and hasattr(ctx, "obj") and ctx.obj:
        ctx.obj.container_tool = tool
    ok = asyncio.run(ctx.obj.run_suite("chaos", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

@sw_app.command(name="integration", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sw_integration(
    ctx: typer.Context,
    tool: str | None = typer.Option(None, "--tool", help="Container tool to use (docker or podman)."),
    # clean: bool = typer.Option(True, "--clean/--no-clean", 
    #     help="Tear down the Docker Compose stack after tests complete. Use --no-clean for debugging."
    # )
) -> None:
    """Tier 5: Heavy Integration (Hashpipe, Static stack)"""
    if tool and hasattr(ctx, "obj") and ctx.obj:
        ctx.obj.container_tool = tool
    ok = asyncio.run(ctx.obj.run_suite("integration", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

@sw_app.command(name="all")
def sw_all(ctx: typer.Context) -> None:
    """Run the full software testing suite (Tiers 1-5)"""
    suites = ["lint", "unit", "logic", "fleet", "chaos", "integration"]
    success = True
    for s in suites:
        ok = asyncio.run(ctx.obj.run_suite(s))
        success = success and ok
    if not success:
        raise typer.Exit(code=1)

@sw_app.command(name="build")
def sw_build(ctx: typer.Context) -> None:
    """Rebuild all test images"""
    asyncio.run(ctx.obj.build_images())

@sw_app.command(name="cleanup")
def sw_cleanup(ctx: typer.Context) -> None:
    """Tear down all test containers and volumes"""
    # TestRunner cleanup logic needed in qa_utils.py
    pass


# ---------------------------------------------------------------------------
# V2 Subcommands (pseti test sw v2 <tier>)
# ---------------------------------------------------------------------------

@v2_app.callback()
def v2_main(
    ctx: typer.Context,
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree for v2 tests.", callback=display_tree_callback)] = False,
) -> None:
    """v2 Software QA — topology-driven, realistic containers."""
    old = ctx.obj
    ctx.obj = TestRunner(V2_QA_TOML_PATH)
    if old:
        ctx.obj.no_teardown = old.no_teardown
        ctx.obj.no_build = old.no_build
        ctx.obj.container_tool = old.container_tool
        ctx.obj.dev_mode = old.dev_mode


@v2_app.command(name="unit", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def v2_unit(ctx: typer.Context) -> None:
    """v2 Tier 1: In-process Pydantic validation and config logic"""
    ok = asyncio.run(ctx.obj.run_suite("unit", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)


@v2_app.command(name="logic", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def v2_logic(ctx: typer.Context) -> None:
    """v2 Tier 2: Subsystem logic with isolated workspace"""
    ok = asyncio.run(ctx.obj.run_suite("logic", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)


@v2_app.command(name="fleet", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def v2_fleet(ctx: typer.Context) -> None:
    """v2 Tier 3: Fleet of sim daqnodes with real gRPC"""
    ok = asyncio.run(ctx.obj.run_suite("fleet", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)


@v2_app.command(name="chaos", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def v2_chaos(ctx: typer.Context) -> None:
    """v2 Tier 4: Fault injection and resilience"""
    ok = asyncio.run(ctx.obj.run_suite("chaos", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)


@v2_app.command(name="integration", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def v2_integration(ctx: typer.Context) -> None:
    """v2 Tier 5: Real hashpipe binary, static stack"""
    ok = asyncio.run(ctx.obj.run_suite("integration", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)


@v2_app.command(name="all")
def v2_all(ctx: typer.Context) -> None:
    """Run all v2 tiers (1-5) sequentially"""
    suites = ["unit", "logic", "fleet", "chaos", "integration"]
    success = True
    for s in suites:
        ok = asyncio.run(ctx.obj.run_suite(s))
        success = success and ok
    if not success:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# SW2 Subcommands (pseti test sw2 <tier>)  — top-level alias for sw v2
# ---------------------------------------------------------------------------

@sw2_app.callback()
def sw2_main(
    ctx: typer.Context,
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree for sw2 tests.", callback=display_tree_callback)] = False,
) -> None:
    """v2 Software QA — topology-driven, realistic containers."""
    old = ctx.obj
    ctx.obj = TestRunner(V2_QA_TOML_PATH)
    if old:
        ctx.obj.no_teardown = old.no_teardown
        ctx.obj.no_build = old.no_build
        ctx.obj.container_tool = old.container_tool
        ctx.obj.dev_mode = old.dev_mode


@sw2_app.command(name="unit", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sw2_unit(ctx: typer.Context) -> None:
    """Tier 1: In-process Pydantic validation and config logic"""
    ok = asyncio.run(ctx.obj.run_suite("unit", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)


@sw2_app.command(name="logic", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sw2_logic(ctx: typer.Context) -> None:
    """Tier 2: Subsystem logic with isolated workspace"""
    ok = asyncio.run(ctx.obj.run_suite("logic", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)


@sw2_app.command(name="fleet", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sw2_fleet(ctx: typer.Context) -> None:
    """Tier 3: Fleet of sim daqnodes with real gRPC"""
    ok = asyncio.run(ctx.obj.run_suite("fleet", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)


@sw2_app.command(name="chaos", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sw2_chaos(ctx: typer.Context) -> None:
    """Tier 4: Fault injection and resilience"""
    ok = asyncio.run(ctx.obj.run_suite("chaos", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)


@sw2_app.command(name="integration", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sw2_integration(ctx: typer.Context) -> None:
    """Tier 5: Real hashpipe binary, static stack"""
    ok = asyncio.run(ctx.obj.run_suite("integration", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)


@sw2_app.command(name="build")
def sw2_build(ctx: typer.Context) -> None:
    """Rebuild all test images"""
    asyncio.run(ctx.obj.build_images())


@sw2_app.command(name="cleanup")
def sw2_cleanup(ctx: typer.Context) -> None:
    """Tear down all test containers and volumes"""
    asyncio.run(ctx.obj.run_suite("cleanup"))


@sw2_app.command(name="all")
def sw2_all(ctx: typer.Context) -> None:
    """Run all v2 tiers (1-5) sequentially"""
    suites = ["unit", "logic", "fleet", "chaos", "integration"]
    success = True
    for s in suites:
        ok = asyncio.run(ctx.obj.run_suite(s))
        success = success and ok

    # Print summary
    from rich.console import Console
    from rich.table import Table
    console = Console()
    table = Table(title="Overall Test Summary (SW v2)")
    table.add_column("Suite", style="cyan")
    table.add_column("Passed", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Skipped", justify="right", style="yellow")
    table.add_column("Error", justify="right", style="red")
    table.add_column("Time (s)", justify="right")

    for res in ctx.obj.all_results:
        st = res.stats
        table.add_row(
            res.name,
            str(st.get("passed", 0)),
            str(st.get("failed", 0)),
            str(st.get("skipped", 0)),
            str(st.get("error", 0)),
            f"{res.elapsed:.1f}"
        )
    console.print(table)

    if not success:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
