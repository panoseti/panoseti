import asyncio
import os
import shutil

import typer

# Heavy imports are moved inside functions to keep pseti startup fast.
from qa_utils import CONTROL_ROOT, QA_TOML_PATH, EnvironmentConfig, SuiteConfig, TestRunner

app = typer.Typer(help="Hardware-Software (HITL) tests", no_args_is_help=True)

@app.callback()
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", "--no-teardown", help="Bypass container teardown for debugging."),
    tool: str = typer.Option("podman", "--tool", help="Container tool to use (docker or podman).")
):
    """
    Hardware-Software (HITL) tests.
    Sets up the TestRunner context and container tool.
    """
    ctx.obj = TestRunner(QA_TOML_PATH)
    ctx.obj.no_teardown = debug
    ctx.obj.container_tool = tool

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
    """Load PANOSETI configs from the HITL-specific directory."""
    from control.utils import config_file, util
    
    # Convert relative to absolute from CONTROL_ROOT
    config_dir = CONTROL_ROOT / env_cfg.config_dir
    
    # Load configs using standard PANOSETI loaders
    daq_cfg = config_file.get_daq_config(dir=str(config_dir))
    net_cfg = config_file.get_network_config(dir=str(config_dir))
    
    # Associate them (merges port forwarding metadata)
    util.attach_daq_config(daq_cfg, net_cfg)
    
    return daq_cfg, net_cfg

def get_ssh_host(node) -> str:
    """Construct ssh:// URI for a node, handling port forwarding gateway."""
    if node.port_forwarding and node.port_forwarding.status:
        # Use gateway IP and custom SSH port
        port = node.port_forwarding.port or 22
        return f"ssh://{node.username}@{node.port_forwarding.gw_ip}:{port}"
    return f"ssh://{node.username}@{node.ip_addr}"

@app.command(name="build")
def hw_build(ctx: typer.Context):
    """Build required container images locally."""
    from rich.console import Console
    console = Console()
    
    runner: TestRunner = ctx.obj
    _suite, env_cfg = get_hw_suite_and_env(ctx)
    tool = runner.container_tool
    
    if tool == "podman":
        console.print("[dim]Ensuring local Podman socket is active...[/dim]")
        asyncio.run(runner._run_cmd("systemctl --user start podman.socket || true"))
    
    cmd = f"{tool} compose -f {CONTROL_ROOT}/{env_cfg.compose_file} --profile headnode --profile daqnode build"
    asyncio.run(runner._run_cmd(cmd))

@app.command()
def check_env(
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

    # Check DAQnode storage
    for node in daq_cfg.daq_nodes:
        console.print(f"[dim]Checking DAQnode ({node.ip_addr}) storage {node.data_dir}...[/dim]")
        
        # Check if remote
        if str(node.ip_addr) != str(daq_cfg.head_node_ip_addr):
            # Remote SSH check
            ssh_host = get_ssh_host(node).replace("ssh://", "") # Convert to standard ssh target
            # Use df to get free space in bytes
            df_cmd = f"ssh {ssh_host} 'df -B1 --output=avail {node.data_dir} | tail -n 1'"
            res = asyncio.run(runner._run_cmd(df_cmd))
            if not res.ok:
                 console.print(f"[red]Error: Could not reach DAQnode {node.ip_addr} via SSH.[/red]")
                 raise typer.Exit(code=1)
        else:
            # Local path already checked or check again if different path
            if not os.path.exists(node.data_dir):
                console.print(f"[red]Error: {node.data_dir} does not exist.[/red]")
                raise typer.Exit(code=1)

    console.print(f"[green]Environment OK. {tool} is ready and space is sufficient.[/green]")

@app.command()
def deploy(ctx: typer.Context):
    """Initialize containers on head node and remote DAQ node."""
    from rich.console import Console
    console = Console()
    
    runner: TestRunner = ctx.obj
    _suite, env_cfg = get_hw_suite_and_env(ctx)
    daq_cfg, _net_cfg = load_hitl_configs(env_cfg)
    tool = runner.container_tool
    
    # Local Headnode
    console.print(f"[cyan]Deploying Headnode profile locally with {tool}...[/cyan]")
    if tool == "podman":
        asyncio.run(runner._run_cmd("systemctl --user start podman.socket || true"))
    
    head_cmd = f"{tool} compose -f {CONTROL_ROOT}/{env_cfg.compose_file} --profile headnode up -d"
    asyncio.run(runner._run_cmd(head_cmd))
    
    # Remote DAQnodes
    for node in daq_cfg.daq_nodes:
        if str(node.ip_addr) == str(daq_cfg.head_node_ip_addr):
            continue
            
        ssh_host = get_ssh_host(node)
        console.print(f"[cyan]Deploying DAQnode profile to {node.ip_addr} via {ssh_host}...[/cyan]")
        
        # Ensure remote socket is active if using podman
        if tool == "podman":
            standard_ssh = ssh_host.replace("ssh://", "")
            remote_init = f"ssh {standard_ssh} 'systemctl --user start podman.socket || true'"
            asyncio.run(runner._run_cmd(remote_init))
            env = {"CONTAINER_HOST": ssh_host, "DOCKER_HOST": ssh_host}
        else:
            env = {"DOCKER_HOST": ssh_host}
            
        daq_cmd = f"{tool} compose -f {CONTROL_ROOT}/{env_cfg.compose_file} --profile daqnode up -d"
        asyncio.run(runner._run_cmd(daq_cmd, env=env))

@app.command()
def clean(ctx: typer.Context):
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
    
    # Remote DAQnodes
    for node in daq_cfg.daq_nodes:
        if str(node.ip_addr) == str(daq_cfg.head_node_ip_addr):
            continue
            
        ssh_host = get_ssh_host(node)
        console.print(f"[yellow]Tearing down DAQnode profile on {node.ip_addr} with {tool}...[/yellow]")
        env = {"CONTAINER_HOST": ssh_host, "DOCKER_HOST": ssh_host} if tool == "podman" else {"DOCKER_HOST": ssh_host}
        
        daq_down = f"{tool} compose -f {CONTROL_ROOT}/{env_cfg.compose_file} --profile daqnode down -v"
        asyncio.run(runner._run_cmd(daq_down, env=env))
    
    console.print(f"[red]Wiping Headnode data {daq_cfg.head_node_data_dir}...[/red]")
    if os.path.exists(daq_cfg.head_node_data_dir):
        asyncio.run(runner._run_cmd(f"sudo rm -rf {daq_cfg.head_node_data_dir}/*"))
    
    for node in daq_cfg.daq_nodes:
        if str(node.ip_addr) != str(daq_cfg.head_node_ip_addr):
             standard_ssh = get_ssh_host(node).replace("ssh://", "")
             console.print(f"[red]Wiping remote data on {node.ip_addr}: {node.data_dir}...[/red]")
             wipe_remote = f"ssh {standard_ssh} 'sudo rm -rf {node.data_dir}/*'"
             asyncio.run(runner._run_cmd(wipe_remote))

@app.command()
def run(ctx: typer.Context):
    """[Placeholder] Run the HW-SW pytest suite."""
    from rich.console import Console
    console = Console()
    console.print("[yellow]HW-SW test suite placeholder. Implementation pending.[/yellow]")

if __name__ == "__main__":
    app()
