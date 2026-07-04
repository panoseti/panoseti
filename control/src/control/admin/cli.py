import asyncio
import os
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from control.utils.paths import PanoPaths

app = typer.Typer(
    help="Admin and deployment tools for remote DAQ nodes.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()

def run_cmd(host: str, cmd: list[str], env: dict[str, str] | None = None) -> bool:
    """Run a shell command and print its output."""
    console.print(f"[[bold cyan]{host}[/bold cyan]] Executing: {' '.join(cmd)}")
    
    # We use subprocess.run so output streams nicely
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        console.print(f"[[bold red]{host}[/bold red]] Command failed with exit code {result.returncode}")
        return False
    console.print(f"[[bold green]{host}[/bold green]] Command succeeded.")
    return True

def get_docker_context_for_node(host: str) -> str:
    from control.utils.config_file import get_daq_config
    daq_config = get_daq_config()
    try:
        node = daq_config.get_node_by_ip(host)
        if node.docker_context:
            return node.docker_context
    except Exception:
        pass
    return f"pseti-daq-{host.replace('.', '-')}"

def resolve_target_nodes(nodes: str) -> list[str]:
    """Expand a comma-separated node list, resolving 'all' from daq_config.json."""
    target_nodes = [n.strip() for n in nodes.split(",")]
    if "all" in target_nodes:
        from control.utils.config_file import get_daq_config
        daq_config = get_daq_config()
        target_nodes = [str(node.ip_addr) for node in daq_config.daq_nodes]
    return target_nodes

async def deploy_node(host: str, mode: str) -> None:
    """Deploy the DAQ node software using the specified strategy."""

    if mode == "docker":
        # We use docker --context to build and deploy natively over SSH
        context = get_docker_context_for_node(host)

        # We assume the context is already created by the user, just like in hw-sw tests.
        # Check if the context exists
        res = subprocess.run(["docker", "context", "ls", "--format", "{{.Name}}"], capture_output=True, text=True)
        if context not in res.stdout:
            console.print(f"[[yellow]{host}[/yellow]] Docker context '{context}' not found. Please create it first:")
            console.print(f"    docker context create {context} --docker \"host=ssh://<user>@{host}\"")
            return

        env = os.environ.copy()
        env["LOCAL_UID"] = str(os.getuid())
        env["LOCAL_GID"] = str(os.getgid())

        compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "docker-compose.daqnode.yml"
        cmd = [
            "docker", "--context", context,
            "compose", "-f", str(compose_file),
            "up", "-d", "--build"
        ]
        run_cmd(host, cmd, env=env)

        # Grafana Alloy (log shipping) is a separate host-network container on the same node.
        alloy_compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "alloy" / "docker-compose.alloy.yml"
        alloy_cmd = [
            "docker", "--context", context,
            "compose", "-f", str(alloy_compose_file),
            "up", "-d", "--build"
        ]
        run_cmd(host, alloy_cmd, env=env)

    elif mode == "bare-metal":
        # For bare-metal, we just SSH in, install from PyPI, and restart the service
        # Ensure we have the SSH key available or it will prompt
        remote_cmd = "source ~/miniconda3/etc/profile.d/conda.sh && conda activate grpc-py314 && pip install --upgrade panoseti-grpc && echo panoseti | sudo -S systemctl restart panoseti_grpc"
        cmd = ["ssh", host, f"bash -c '{remote_cmd}'"]
        run_cmd(host, cmd)

    else:
        console.print(f"[red]Unknown deployment mode: {mode}[/red]")

@app.command()
def deploy(
    nodes: Annotated[str, typer.Argument(help="Comma-separated list of hostnames/IPs, or 'all'.")],
    mode: Annotated[str, typer.Option("--mode", help="'docker' or 'bare-metal' deployment strategy.")] = "docker"
) -> None:
    """Deploy the DAQ node gRPC and telemetry stack to remote machines."""
    if mode not in ["docker", "bare-metal"]:
        console.print("[bold red]Error:[/] --mode must be either 'docker' or 'bare-metal'.")
        raise typer.Exit(1)
        
    target_nodes = resolve_target_nodes(nodes)

    console.print(f"[bold]Starting {mode} deployment on: {', '.join(target_nodes)}[/bold]")
    
    # Run sequentially or concurrently? Subprocess stdout interleaves if concurrent.
    # We will just run them sequentially for clear logs since we use subprocess.run directly.
    for host in target_nodes:
        asyncio.run(deploy_node(host, mode))


@app.command()
def status(
    nodes: Annotated[str, typer.Argument(help="Comma-separated list of hostnames/IPs, or 'all'.")],
    mode: Annotated[str, typer.Option("--mode", help="'docker' or 'bare-metal' deployment strategy.")] = "docker"
) -> None:
    """Check the status of the DAQ node services."""
    target_nodes = resolve_target_nodes(nodes)

    for host in target_nodes:
        if mode == "docker":
            context = get_docker_context_for_node(host)
            compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "docker-compose.daqnode.yml"
            cmd = ["docker", "--context", context, "compose", "-f", str(compose_file), "ps"]
            run_cmd(host, cmd)

            alloy_compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "alloy" / "docker-compose.alloy.yml"
            alloy_cmd = ["docker", "--context", context, "compose", "-f", str(alloy_compose_file), "ps"]
            run_cmd(host, alloy_cmd)
        else:
            cmd = ["ssh", host, "systemctl is-active panoseti_grpc panoseti_alloy"]
            run_cmd(host, cmd)

if __name__ == "__main__":
    app()
