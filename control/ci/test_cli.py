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
from typing import Annotated, Any, Optional

import typer

from qa_utils import (
    CONTROL_ROOT,
    QA_TOML_PATH,
    EnvironmentConfig,
    SSHTunnel,
    SuiteConfig,
    TestRunner,
)

app = typer.Typer(
    help="PSETI Quality Assurance & Testing Suite.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Sub-apps for organization
sw_app = typer.Typer(help="Software QA tests (Docker-based CI simulations)", no_args_is_help=True)
hw_app = typer.Typer(help="Hardware-in-the-Loop (HITL) physical lab tests", no_args_is_help=True)
lint_app = typer.Typer(help="Static analysis and linting (Ruff, MyPy)", no_args_is_help=True)

app.add_typer(sw_app, name="sw")
app.add_typer(hw_app, name="hw")
app.add_typer(lint_app, name="lint")

# ---------------------------------------------------------------------------
# Global Setup
# ---------------------------------------------------------------------------

@app.callback()
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", "--no-teardown", help="Bypass container teardown for debugging."),
    no_build: bool = typer.Option(False, "--no-build", help="Do not attempt to build images, use existing ones."),
    tool: str = typer.Option("docker", "--tool", help="Container tool to use (docker or podman).")
):
    """
    PSETI Testing Suite.
    """
    ctx.obj = TestRunner(QA_TOML_PATH)
    ctx.obj.no_teardown = debug
    ctx.obj.no_build = no_build
    ctx.obj.container_tool = tool


# ---------------------------------------------------------------------------
# LINT Subcommands
# ---------------------------------------------------------------------------

@lint_app.callback(invoke_without_command=True)
def lint_main(
    ctx: typer.Context,
    targets: Annotated[str, typer.Argument(help="Scope to lint: 'ruff', 'mypy', or 'all'")] = "all",
):
    """Run linters [ruff/mypy args...]"""
    if ctx.invoked_subcommand is not None:
        return
    ok = asyncio.run(ctx.obj.run_suite("lint", target=targets, extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# SW Subcommands (Software QA)
# ---------------------------------------------------------------------------

@sw_app.command(name="unit", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sw_unit(ctx: typer.Context, jobs: int | None = typer.Option(None, "--jobs", "-j", help="Parallel jobs")):
    """Run unit tests [-j N] [pytest args...]"""
    ok = asyncio.run(ctx.obj.run_suite("unit", jobs=jobs, extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

@sw_app.command(name="integration", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sw_integration(ctx: typer.Context):
    """Run integration tests [pytest args...]"""
    ok = asyncio.run(ctx.obj.run_suite("integration", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

@sw_app.command(name="structural", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sw_structural(ctx: typer.Context):
    """Run structural/topology tests [pytest args...]"""
    ok = asyncio.run(ctx.obj.run_suite("structural", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

@sw_app.command(name="chaos", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sw_chaos(ctx: typer.Context):
    """Run chaos scenario tests [pytest args...]"""
    ok = asyncio.run(ctx.obj.run_suite("chaos", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

@sw_app.command(name="all")
def sw_all(ctx: typer.Context):
    """Run the full software testing suite (unit, structural, integration)"""
    suites = ["unit", "structural", "integration"]
    success = True
    for s in suites:
        ok = asyncio.run(ctx.obj.run_suite(s))
        success = success and ok
    if not success:
        raise typer.Exit(code=1)

@sw_app.command(name="build")
def sw_build(ctx: typer.Context):
    """Rebuild all test images"""
    asyncio.run(ctx.obj.build_all())

@sw_app.command(name="cleanup")
def sw_cleanup(ctx: typer.Context):
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
    util.attach_daq_config(daq_cfg, net_cfg)
    return daq_cfg, net_cfg

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
def hw_build(ctx: typer.Context):
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
):
    """Verify environment, disk space, and container engine."""
    from rich.console import Console
    console = Console()
    runner: TestRunner = ctx.obj
    _suite, env_cfg = get_hw_suite_and_env(ctx)
    daq_cfg, _net_cfg = load_hitl_configs(env_cfg)
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
def hw_deploy(ctx: typer.Context):
    """Initialize containers on head node and remote DAQ node."""
    from rich.console import Console
    console = Console()
    runner: TestRunner = ctx.obj
    _suite, env_cfg = get_hw_suite_and_env(ctx)
    daq_cfg, _net_cfg = load_hitl_configs(env_cfg)
    tool = runner.container_tool
    
    console.print(f"[cyan]Deploying Headnode profile locally with {tool}...[/cyan]")
    if tool == "podman":
        asyncio.run(runner._run_cmd("systemctl --user start podman.socket || true"))
    
    head_cmd = f"{tool} compose -f {CONTROL_ROOT}/{env_cfg.compose_file} --profile headnode up -d"
    if runner.no_build: head_cmd += " --no-build"
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
            daq_cmd = f"{tool} compose -f {CONTROL_ROOT}/{env_cfg.compose_file} --profile daqnode up -d"
            if runner.no_build: daq_cmd += " --no-build"
            asyncio.run(runner._run_cmd(daq_cmd, env=env))

@hw_app.command(name="clean")
def hw_clean(ctx: typer.Context):
    """Tear down containers and wipe physical data directory."""
    from rich.console import Console
    console = Console()
    runner: TestRunner = ctx.obj
    _suite, env_cfg = get_hw_suite_and_env(ctx)
    daq_cfg, _net_cfg = load_hitl_configs(env_cfg)
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
    console.print(f"[yellow]Placeholder: Data wiping skipped.[/yellow]")

@hw_app.command(name="run")
def hw_run(ctx: typer.Context):
    """[Placeholder] Run the HW-SW pytest suite."""
    from rich.console import Console
    Console().print("[yellow]HW-SW test suite placeholder. Implementation pending.[/yellow]")

if __name__ == "__main__":
    app()
