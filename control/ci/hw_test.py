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
        port = node.port_forwarding.port or 22
        # Omit port 22 to avoid Docker Compose URI parsing bugs
        if port == 22:
            return f"ssh://{node.username}@{node.port_forwarding.gw_ip}"
        return f"ssh://{node.username}@{node.port_forwarding.gw_ip}:{port}"
    return f"ssh://{node.username}@{node.ip_addr}"

def get_raw_ssh_args(ssh_host_uri: str) -> str:
    """Convert a ssh:// URI into raw ssh command arguments (e.g., '-p port target')."""
    uri = ssh_host_uri.replace("ssh://", "")
    if ":" in uri:
        target, port = uri.split(":")
        return f"-p {port} {target}"
    return uri

async def resolve_remote_podman_uri(runner: TestRunner, ssh_host_uri: str) -> str:
    """
    Connects to the remote host to find the correct rootless Podman socket path.
    Returns a full URI like ssh://user@host:port/run/user/UID/podman/podman.sock
    """
    ssh_args = get_raw_ssh_args(ssh_host_uri)
    
    # 1. Ensure the socket is active (also triggers systemd user manager if needed)
    await runner._run_cmd(f"ssh {ssh_args} 'systemctl --user start podman.socket || true'", quiet=True)
    
    # 2. Query remote UID and socket existence
    # We check for the rootless socket first, then fall back to the root socket.
    remote_cmd = (
        "id -u && "
        "if [ -S /run/user/$(id -u)/podman/podman.sock ]; then echo /run/user/$(id -u)/podman/podman.sock; "
        "elif [ -S /run/podman/podman.sock ]; then echo /run/podman/podman.sock; "
        "else echo AUTO; fi"
    )
    
    res = await runner._run_cmd(f"ssh {ssh_args} '{remote_cmd}'", capture=True)
    if res.ok and res.stdout:
        lines = res.stdout.strip().splitlines()
        if len(lines) >= 2:
            _uid = lines[0]
            path = lines[1]
            if path != "AUTO":
                # Construct URI with path. 
                # Docker/Podman URI format: ssh://user@host[:port]/path/to/socket
                return f"{ssh_host_uri}{path}"
                
    return ssh_host_uri

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
            ssh_args = get_raw_ssh_args(get_ssh_host(node))
            # Use df to get free space in bytes
            df_cmd = f"ssh {ssh_args} 'df -B1 --output=avail {node.data_dir} | tail -n 1'"
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
        
        # Determine remote URI (handle podman socket path)
        if tool == "podman":
            resolved_uri = asyncio.run(resolve_remote_podman_uri(runner, ssh_host))
            if resolved_uri != ssh_host:
                console.print(f"[dim]Resolved remote Podman socket: {resolved_uri}[/dim]")
            env = {"CONTAINER_HOST": resolved_uri, "DOCKER_HOST": resolved_uri}
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
        
        if tool == "podman":
             resolved_uri = asyncio.run(resolve_remote_podman_uri(runner, ssh_host))
             env = {"CONTAINER_HOST": resolved_uri, "DOCKER_HOST": resolved_uri}
        else:
             env = {"DOCKER_HOST": ssh_host}
        
        daq_down = f"{tool} compose -f {CONTROL_ROOT}/{env_cfg.compose_file} --profile daqnode down -v"
        asyncio.run(runner._run_cmd(daq_down, env=env))
    
    # console.print(f"[red]Wiping Headnode data {daq_cfg.head_node_data_dir}...[/red]")
    # if os.path.exists(daq_cfg.head_node_data_dir):
    #     asyncio.run(runner._run_cmd(f"rm -rf {daq_cfg.head_node_data_dir}/*"))
    console.print(f"[yellow]Placeholder: Headnode data wiping skipped for {daq_cfg.head_node_data_dir}[/yellow]")
    
    for node in daq_cfg.daq_nodes:
        if str(node.ip_addr) != str(daq_cfg.head_node_ip_addr):
             # ssh_args = get_raw_ssh_args(get_ssh_host(node))
             # console.print(f"[red]Wiping remote data on {node.ip_addr}: {node.data_dir}...[/red]")
             # wipe_remote = f"ssh {ssh_args} 'rm -rf {node.data_dir}/*'"
             # asyncio.run(runner._run_cmd(wipe_remote))
             console.print(f"[yellow]Placeholder: Remote data wiping skipped for {node.ip_addr}:{node.data_dir}[/yellow]")

@app.command()
def run(ctx: typer.Context):
    """[Placeholder] Run the HW-SW pytest suite."""
    from rich.console import Console
    console = Console()
    console.print("[yellow]HW-SW test suite placeholder. Implementation pending.[/yellow]")

if __name__ == "__main__":
    app()
