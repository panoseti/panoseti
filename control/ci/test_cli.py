"""
Unified test orchestration CLI for PSETI.
Provides subcommands for Linting, Software (Docker CI), and Hardware (HITL) tests.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Annotated, Any

import click
import typer
import typer.core
from panoseti_grpc.util.cli import BaseLazyGroup, display_tree_callback
from qa_utils import (
    CONTROL_ROOT,
    QA_TOML_PATH,
    EnvironmentConfig,
    SSHTunnel,
    SuiteConfig,
    TestRunner,
)


class GrpcTestLazyGroup(BaseLazyGroup):
    """
    Lazy-loading group for gRPC service layer tests.
    Unwraps the tests.qa app from the grpc/ directory.
    """
    def list_commands(self, ctx: click.Context) -> list[str]:
        import importlib
        root = Path(__file__).parent.parent.parent
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

    def get_command(self, ctx: click.Context, name: str) -> click.Command | None:
        import importlib
        root = Path(__file__).parent.parent.parent
        grpc_tests = str(root / "grpc")
        if Path(grpc_tests).exists() and grpc_tests not in sys.path:
            sys.path.insert(0, grpc_tests)
            
        try:
            mod = importlib.import_module("tests.qa")
            test_app = mod.app
            click_group = typer.main.get_command(test_app)
            cmd = click_group.get_command(ctx, name)  # type: ignore[attr-defined]
            if cmd:
                cmd.name = name
                return cmd
            return None
        except Exception as e:
            click.secho(f"Error loading gRPC test command '{name}': {e}", fg="red", err=True)
            return None


app = typer.Typer(
    help="PSETI Quality Assurance & Testing Suite.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Sub-apps for organization
sw_app = typer.Typer(help="Software QA tests (Docker-based CI simulations)", no_args_is_help=True)
hw_app = typer.Typer(help="Hardware-in-the-Loop (HITL) physical lab tests", no_args_is_help=True)
# lint_app was removed
# The grpc sub-app uses a special lazy group to map to tests.qa:app
grpc_app = typer.Typer(help="gRPC service layer tests", no_args_is_help=True, cls=GrpcTestLazyGroup)

app.add_typer(sw_app, name="sw")
app.add_typer(hw_app, name="hw")
# app.add_typer(lint_app, name="lint") removed
app.add_typer(grpc_app, name="grpc")

@grpc_app.callback()
def grpc_main(
    ctx: typer.Context,
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree for gRPC tests.", callback=display_tree_callback)] = False
) -> None:
    """gRPC service layer tests"""
    if tree:
        return
    # Ensure we are running from grpc/tests so that relative paths in qa.toml resolve
    root = Path(__file__).parent.parent.parent
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

@app.callback()
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", "--no-teardown", help="Bypass container teardown for debugging."),
    no_build: bool = typer.Option(False, "--no-build", help="Do not attempt to build images, use existing ones."),
    tool: str = typer.Option("docker", "--tool", help="Container tool to use (docker or podman)."),
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree for PSETI testing.", callback=display_tree_callback)] = False
) -> None:
    """
    PSETI Testing Suite.
    """
    ctx.obj = TestRunner(QA_TOML_PATH)
    ctx.obj.no_teardown = debug
    ctx.obj.no_build = no_build
    ctx.obj.container_tool = tool

@sw_app.callback()
def sw_main(
    ctx: typer.Context,
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree for software tests.", callback=display_tree_callback)] = False
) -> None:
    """Software QA tests (Docker-based CI simulations)"""
    pass

@hw_app.callback()
def hw_main(
    ctx: typer.Context,
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree for hardware tests.", callback=display_tree_callback)] = False
) -> None:
    """Hardware-in-the-Loop (HITL) physical lab tests"""
    pass

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
def sw_unit(ctx: typer.Context, jobs: int | None = typer.Option(None, "--jobs", "-j", help="Parallel jobs")) -> None:
    """Run unit tests [-j N] [pytest args...]"""
    ok = asyncio.run(ctx.obj.run_suite("unit", jobs=jobs, extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

@sw_app.command(name="integration", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sw_integration(ctx: typer.Context) -> None:
    """Run integration tests [pytest args...]"""
    ok = asyncio.run(ctx.obj.run_suite("integration", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

@sw_app.command(name="structural", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sw_structural(ctx: typer.Context) -> None:
    """Run structural/topology tests [pytest args...]"""
    ok = asyncio.run(ctx.obj.run_suite("structural", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

@sw_app.command(name="chaos", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sw_chaos(ctx: typer.Context) -> None:
    """Run chaos scenario tests [pytest args...]"""
    ok = asyncio.run(ctx.obj.run_suite("chaos", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

@sw_app.command(name="all")
def sw_all(ctx: typer.Context) -> None:
    """Run the full software testing suite (unit, structural, integration)"""
    suites = ["lint", "unit", "structural", "integration", "chaos"]
    success = True
    for s in suites:
        ok = asyncio.run(ctx.obj.run_suite(s))
        success = success and ok
    if not success:
        raise typer.Exit(code=1)

@sw_app.command(name="build")
def sw_build(ctx: typer.Context) -> None:
    """Rebuild all test images"""
    asyncio.run(ctx.obj.build_all())

@sw_app.command(name="cleanup")
def sw_cleanup(ctx: typer.Context) -> None:
    """Tear down all test containers and volumes"""
    asyncio.run(ctx.obj.cleanup_all())


# ---------------------------------------------------------------------------
# HW Subcommands (HITL)
# ---------------------------------------------------------------------------

def get_hw_suite_and_env(ctx: typer.Context) -> tuple[SuiteConfig, EnvironmentConfig]:
    """Helper to resolve the hardware suite and its environment configuration."""
    from rich.console import Console
    console = Console()
    
    runner: TestRunner = ctx.obj
    suite_name = "test-hw"
    if suite_name not in runner.cfg.suites:
        console.print(f"[red]Error: Suite {suite_name} not found in qa.toml[/red]")
        raise typer.Exit(code=1)
    
    suite = runner.cfg.suites[suite_name]
    if not suite.environment:
        console.print(f"[red]Error: Suite {suite_name} does not specify an environment[/red]")
        raise typer.Exit(code=1)
    
    if suite.environment not in runner.cfg.environments:
        console.print(f"[red]Error: Environment {suite.environment} not found in qa.toml[/red]")
        raise typer.Exit(code=1)
    
    return suite, runner.cfg.environments[suite.environment]

def load_hitl_configs(env_cfg: EnvironmentConfig):
    """Load PSETI configs from the HITL-specific directory."""
    from control.utils import config_file, util
    config_dir = CONTROL_ROOT / env_cfg.config_dir
    daq_cfg = config_file.get_daq_config(dir=str(config_dir))
    net_cfg = config_file.get_network_config(dir=str(config_dir))
    obs_cfg = config_file.get_obs_config(dir=str(config_dir))
    util.attach_daq_config(daq_cfg, net_cfg)
    return daq_cfg, net_cfg, obs_cfg

def get_ssh_host(node: Any) -> str:
    """Construct ssh:// URI for a node, handling port forwarding gateway."""
    if node.port_forwarding and node.port_forwarding.status:
        port = node.port_forwarding.port or 22
        if port == 22:
            return f"ssh://{node.username}@{node.port_forwarding.gw_ip}"
        return f"ssh://{node.username}@{node.port_forwarding.gw_ip}:{port}"
    return f"ssh://{node.username}@{node.ip_addr}"

def get_raw_ssh_args(ssh_host_uri: str) -> str:
    """Convert a ssh:// URI into raw ssh command arguments."""
    uri = ssh_host_uri.replace("ssh://", "")
    if ":" in uri:
        target, port = uri.split(":")
        return f"-p {port} {target}"
    return uri

async def resolve_remote_socket_path(runner: TestRunner, ssh_args: str) -> str:
    """Query remote host to find the correct rootless Podman socket path."""
    await runner._run_cmd(f"ssh -o BatchMode=yes {ssh_args} 'systemctl --user start podman.socket || true'", quiet=True)
    remote_cmd = (
        "if [ -S /run/user/$(id -u)/podman/podman.sock ]; then echo /run/user/$(id -u)/podman/podman.sock; "
        "elif [ -S /run/podman/podman.sock ]; then echo /run/podman/podman.sock; "
        "else echo /run/podman/podman.sock; fi"
    )
    res = await runner._run_cmd(f"ssh -o BatchMode=yes {ssh_args} '{remote_cmd}'", capture=True)
    return res.stdout.strip() if res.ok and res.stdout else "/run/podman/podman.sock"

@hw_app.command(name="build")
def hw_build(ctx: typer.Context) -> None:
    """Build required container images locally."""
    from rich.console import Console
    console = Console()
    runner: TestRunner = ctx.obj
    _suite, env_cfg = get_hw_suite_and_env(ctx)
    tool = runner.container_tool
    env = {"BUILDAH_ISOLATION": "chroot"} if tool == "podman" else {}
    if tool == "podman":
        console.print("[dim]Ensuring local Podman socket is active...[/dim]")
        asyncio.run(runner._run_cmd("systemctl --user start podman.socket || true"))
    cmd = f"{tool} compose -f {CONTROL_ROOT}/{env_cfg.compose_file} --profile headnode --profile daqnode build"
    asyncio.run(runner._run_cmd(cmd, env=env))

@hw_app.command(name="check-env")
def hw_check_env(
    ctx: typer.Context, 
    min_gb: int = typer.Option(10, "--min-gb", help="Minimum required free space in GB.")
) -> None:
    """Verify environment, disk space, and container engine."""
    from rich.console import Console
    console = Console()
    runner: TestRunner = ctx.obj
    _suite, env_cfg = get_hw_suite_and_env(ctx)
    daq_cfg, _net_cfg, obs_cfg = load_hitl_configs(env_cfg)
    tool = runner.container_tool
    
    console.print(f"[dim]Checking container engine ({tool})...[/dim]")
    res = asyncio.run(runner._run_cmd(f"{tool} version", quiet=True))
    if not res.ok:
        console.print(f"[red]Error: {tool} is not installed or responsive.[/red]")
        raise typer.Exit(code=1)
        
    if tool == "podman":
        res = asyncio.run(runner._run_cmd("systemctl --user is-active podman.socket", quiet=True))
        if not res.ok:
            console.print("[yellow]Warning: podman.socket is not active. Attempting to start...[/yellow]")
            asyncio.run(runner._run_cmd("systemctl --user start podman.socket"))

    # Check Headnode storage
    console.print(f"[dim]Checking Headnode storage {daq_cfg.head_node_data_dir}...[/dim]")
    if not os.path.exists(daq_cfg.head_node_data_dir):
        console.print(f"[red]Error: {daq_cfg.head_node_data_dir} does not exist.[/red]")
        raise typer.Exit(code=1)
    
    usage = shutil.disk_usage(daq_cfg.head_node_data_dir)
    free_gb = usage.free / (2**30)
    if free_gb < min_gb:
        console.print(f"[red]Error: Low Headnode space. {free_gb:.1f}GB free, {min_gb}GB required.[/red]")
        raise typer.Exit(code=1)

    # Check WPS reachability
    console.print("[dim]Checking WPS power supplies reachability...[/dim]")
    extra_data = obs_cfg.model_extra or {}
    for key, val in extra_data.items():
        if key.startswith("wps"):
            url = val.get("url")
            if url:
                console.print(f"[dim]Checking WPS {key} at {url}...[/dim]")
                # Use a short timeout for the reachability check
                cmd = f"curl -s --connect-timeout 2 --head {url}"
                res = asyncio.run(runner._run_cmd(cmd, quiet=True))
                if res.ok:
                    console.print(f"[green]✔ WPS {key} is reachable.[/green]")
                else:
                    console.print(f"[yellow]⚠ WPS {key} is NOT reachable (url={url}).[/yellow]")

    # Check remote connectivity and DAQnode storage
    for node in daq_cfg.daq_nodes:
        if str(node.ip_addr) == str(daq_cfg.head_node_ip_addr):
            continue

        console.print(f"[dim]Checking SSH credentials for {node.ip_addr}...[/dim]")
        ssh_args = get_raw_ssh_args(get_ssh_host(node))
        res = asyncio.run(runner._run_cmd(f"ssh -o BatchMode=yes {ssh_args} 'true'", quiet=True))
        if not res.ok:
             console.print(f"[red]Error: SSH key-based authentication failed for {node.ip_addr}.[/red]")
             raise typer.Exit(code=1)

        console.print(f"[dim]Checking DAQnode ({node.ip_addr}) storage {node.data_dir}...[/dim]")
        df_cmd = f"ssh {ssh_args} 'df -B1 --output=avail {node.data_dir} | tail -n 1'"
        res = asyncio.run(runner._run_cmd(df_cmd, capture=True))
        if res.ok and res.stdout:
            try:
                free_bytes = int(res.stdout.strip())
                node_free_gb = free_bytes / (2**30)
                if node_free_gb < min_gb:
                    console.print(f"[red]Error: Low space on {node.ip_addr}. {node_free_gb:.1f}GB free.[/red]")
                    raise typer.Exit(code=1)
            except ValueError:
                console.print(f"[red]Error: Could not parse disk space from {node.ip_addr}.[/red]")
                raise typer.Exit(code=1) from None

    console.print(f"[green]Environment OK. {tool} is ready and space is sufficient.[/green]")

@hw_app.command(name="deploy")
def hw_deploy(ctx: typer.Context) -> None:
    """Initialize containers on head node and remote DAQ node."""
    from rich.console import Console
    console = Console()
    runner: TestRunner = ctx.obj
    _suite, env_cfg = get_hw_suite_and_env(ctx)
    daq_cfg, _net_cfg, _obs_cfg = load_hitl_configs(env_cfg)
    tool = runner.container_tool
    
    console.print(f"[cyan]Deploying Headnode profile locally with {tool}...[/cyan]")
    if tool == "podman":
        asyncio.run(runner._run_cmd("systemctl --user start podman.socket || true"))
    
    head_cmd = f"{tool} compose -f {CONTROL_ROOT}/{env_cfg.compose_file} --profile headnode up -d"
    if runner.no_build:
        head_cmd += " --no-build"
    asyncio.run(runner._run_cmd(head_cmd))
    
    for node in daq_cfg.daq_nodes:
        if str(node.ip_addr) == str(daq_cfg.head_node_ip_addr):
            continue
        ssh_host_uri = get_ssh_host(node)
        ssh_args = get_raw_ssh_args(ssh_host_uri)
        console.print(f"[cyan]Deploying DAQnode profile to {node.ip_addr} via tunnel...[/cyan]")
        remote_sock = asyncio.run(resolve_remote_socket_path(runner, ssh_args))
        with SSHTunnel(ssh_args, remote_sock) as local_sock:
            env = {"CONTAINER_HOST": f"unix://{local_sock}", "DOCKER_HOST": f"unix://{local_sock}"}
            
            if not runner.no_build:
                console.print(f"[cyan]Transferring pseti-daqnode:hitl image to {node.ip_addr}...[/cyan]")
                transfer_cmd = f"{tool} save pseti-daqnode:hitl | env DOCKER_HOST=unix://{local_sock} CONTAINER_HOST=unix://{local_sock} {tool} load"
                os.system(transfer_cmd)
                
            daq_cmd = f"{tool} compose -f {CONTROL_ROOT}/{env_cfg.compose_file} --profile daqnode up -d"
            if runner.no_build:
                daq_cmd += " --no-build"
            asyncio.run(runner._run_cmd(daq_cmd, env=env))

@hw_app.command(name="clean")
def hw_clean(ctx: typer.Context) -> None:
    """Tear down containers and wipe physical data directory."""
    from rich.console import Console
    console = Console()
    runner: TestRunner = ctx.obj
    _suite, env_cfg = get_hw_suite_and_env(ctx)
    daq_cfg, _net_cfg, _obs_cfg = load_hitl_configs(env_cfg)
    tool = runner.container_tool
    
    console.print(f"[yellow]Tearing down Headnode profile with {tool}...[/yellow]")
    head_down = f"{tool} compose -f {CONTROL_ROOT}/{env_cfg.compose_file} --profile headnode down -v"
    asyncio.run(runner._run_cmd(head_down))
    
    for node in daq_cfg.daq_nodes:
        if str(node.ip_addr) == str(daq_cfg.head_node_ip_addr):
            continue
        ssh_host_uri = get_ssh_host(node)
        ssh_args = get_raw_ssh_args(ssh_host_uri)
        console.print(f"[yellow]Tearing down DAQnode profile on {node.ip_addr} via tunnel...[/yellow]")
        remote_sock = asyncio.run(resolve_remote_socket_path(runner, ssh_args))
        with SSHTunnel(ssh_args, remote_sock) as local_sock:
            env = {"CONTAINER_HOST": f"unix://{local_sock}", "DOCKER_HOST": f"unix://{local_sock}"}
            daq_down = f"{tool} compose -f {CONTROL_ROOT}/{env_cfg.compose_file} --profile daqnode down -v"
            asyncio.run(runner._run_cmd(daq_down, env=env))
    console.print("[yellow]Placeholder: Data wiping skipped.[/yellow]")

@hw_app.command(name="attach")
def hw_attach(ctx: typer.Context) -> None:
    """Enter the headnode container shell for debugging."""
    from rich.console import Console
    console = Console()
    runner: TestRunner = ctx.obj
    _suite, env_cfg = get_hw_suite_and_env(ctx)
    tool = runner.container_tool
    
    cmd = f"{tool} compose -f {CONTROL_ROOT}/{env_cfg.compose_file} --profile headnode exec headnode-server /bin/bash"
    console.print("[cyan]Attaching to headnode-server...[/cyan]")
    os.system(cmd)

@hw_app.command(name="run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def hw_run(ctx: typer.Context) -> None:
    """Run the physical hardware-software (HITL) test suite."""
    ok = asyncio.run(ctx.obj.run_suite("test-hw", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
