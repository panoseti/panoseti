import asyncio
import os
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from control.utils.paths import PanoPaths

app = typer.Typer(
    help="Admin and deployment tools for remote DAQ nodes and the head node.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()

# Env vars whose values actually change compose interpolation/behavior --
# printed alongside every command so it can be copy-pasted and re-run (or
# tweaked) verbatim, without having to reconstruct what this process
# resolved them to.
_PRINTABLE_ENV_KEYS = (
    "PSETI_ROOT_BUILD", "PSETI_CONFIG", "PSETI_DATA_DIR", "DAQ_DATA_DIR",
    "HEADNODE_IP", "HOST_UID", "HOST_GID", "LOCAL_UID", "LOCAL_GID",
)


def run_cmd(host: str, cmd: list[str], env: dict[str, str] | None = None) -> bool:
    """Run a shell command, printing the full reproducible invocation first."""
    if env:
        shown = " ".join(f"{k}={env[k]}" for k in _PRINTABLE_ENV_KEYS if k in env)
        if shown:
            console.print(f"[[bold cyan]{host}[/bold cyan]] {shown} \\")
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

def resolve_target_nodes(nodes: str) -> tuple[list[str], bool]:
    """Expand a comma-separated node list, resolving 'all' from daq_config.json.

    Returns (daq_node_targets, include_headnode). 'headnode' is not a DAQ
    node IP -- it's pulled out and returned separately since it deploys
    locally with no docker context / SSH involved. 'all' means every DAQ
    node *and* the head node.
    """
    target_nodes = [n.strip() for n in nodes.split(",")]
    if "all" in target_nodes:
        from control.utils.config_file import get_daq_config
        daq_config = get_daq_config()
        return [str(node.ip_addr) for node in daq_config.daq_nodes], True

    include_headnode = "headnode" in target_nodes
    daq_targets = [n for n in target_nodes if n != "headnode"]
    return daq_targets, include_headnode


def get_headnode_compose_env() -> dict[str, str] | None:
    """Build the env dict required by control/deploy/docker-compose.headnode.yml.

    Returns None (after printing why) if a value that has no safe default
    -- HEADNODE_IP -- isn't set anywhere.
    """
    env = os.environ.copy()
    env["PSETI_ROOT_BUILD"] = str(PanoPaths.software_root_dir())
    env.setdefault("PSETI_CONFIG", str(PanoPaths.config_dir()))
    env.setdefault("PSETI_DATA_DIR", "/mnt/panoseti-data")
    env["HOST_UID"] = str(os.getuid())
    env["HOST_GID"] = str(os.getgid())
    if "HEADNODE_IP" not in env:
        console.print(
            "[bold red][headnode][/bold red] HEADNODE_IP is not set (this machine's "
            "real IP -- required so Alloy knows where to push logs). Set it and retry, e.g.:\n"
            "    HEADNODE_IP=192.168.88.103 pseti admin deploy headnode"
        )
        return None
    return env


def deploy_headnode(mode: str) -> bool:
    """Deploy the head node's observability + gRPC gateway stack (local machine, no SSH)."""
    if mode != "docker":
        console.print(f"[yellow][headnode][/yellow] --mode {mode} is not supported for the head node; use --mode docker.")
        return False

    env = get_headnode_compose_env()
    if env is None:
        return False

    compose_file = PanoPaths.base_dir() / "deploy" / "docker-compose.headnode.yml"
    cmd = ["docker", "compose", "-p", "pseti-headnode", "-f", str(compose_file), "up", "-d", "--build"]
    return run_cmd("headnode", cmd, env=env)


def status_headnode(mode: str) -> None:
    """Check the status of the head node's local compose stack."""
    if mode != "docker":
        console.print(f"[yellow][headnode][/yellow] --mode {mode} is not supported for the head node; use --mode docker.")
        return

    env = get_headnode_compose_env()
    if env is None:
        return

    compose_file = PanoPaths.base_dir() / "deploy" / "docker-compose.headnode.yml"
    cmd = ["docker", "compose", "-p", "pseti-headnode", "-f", str(compose_file), "ps"]
    run_cmd("headnode", cmd, env=env)

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

        project_name = f"pseti-daqnode-{host.replace('.', '-')}"

        compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "docker-compose.daqnode.yml"
        cmd = [
            "docker", "--context", context,
            "compose", "-p", project_name, "-f", str(compose_file),
            "up", "-d", "--build"
        ]
        run_cmd(host, cmd, env=env)

        # Grafana Alloy (log shipping) is a separate host-network container on the same node.
        # Skip if this DAQ node is the head node (headnode-server stack already runs it).
        from control.utils.util import is_local
        from control.utils.config_file import get_daq_config
        is_headnode = is_local(host, get_daq_config())
        if not is_headnode:
            alloy_compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "alloy" / "docker-compose.alloy.yml"
            alloy_cmd = [
                "docker", "--context", context,
                "compose", "-p", project_name, "-f", str(alloy_compose_file),
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
    nodes: Annotated[str, typer.Argument(help="Comma-separated list of hostnames/IPs, 'headnode', or 'all' (every DAQ node + the head node).")],
    mode: Annotated[str, typer.Option("--mode", help="'docker' or 'bare-metal' deployment strategy.")] = "docker"
) -> None:
    """Deploy the DAQ node gRPC/telemetry stack and/or the head node stack."""
    if mode not in ["docker", "bare-metal"]:
        console.print("[bold red]Error:[/] --mode must be either 'docker' or 'bare-metal'.")
        raise typer.Exit(1)

    daq_targets, include_headnode = resolve_target_nodes(nodes)

    described = (["headnode"] if include_headnode else []) + daq_targets
    console.print(f"[bold]Starting {mode} deployment on: {', '.join(described)}[/bold]")

    # Run sequentially: subprocess stdout interleaves if concurrent, and we
    # print each full invocation as it runs so the log doubles as a
    # re-runnable/modifiable script.
    if include_headnode:
        deploy_headnode(mode)

    for host in daq_targets:
        asyncio.run(deploy_node(host, mode))


@app.command()
def build(
    nodes: Annotated[str, typer.Argument(help="Comma-separated list of hostnames/IPs, 'headnode', or 'all' (every DAQ node + the head node).")],
    mode: Annotated[str, typer.Option("--mode", help="'docker' or 'bare-metal' deployment strategy.")] = "docker"
) -> None:
    """Build the DAQ node gRPC/telemetry stack and/or the head node stack images."""
    if mode != "docker":
        console.print("[bold red]Error:[/] build is only supported in docker mode.")
        raise typer.Exit(1)

    daq_targets, include_headnode = resolve_target_nodes(nodes)

    if include_headnode:
        env = get_headnode_compose_env()
        if env is not None:
            compose_file = PanoPaths.base_dir() / "deploy" / "docker-compose.headnode.yml"
            cmd = ["docker", "compose", "-p", "pseti-headnode", "-f", str(compose_file), "build"]
            run_cmd("headnode", cmd, env=env)

    for host in daq_targets:
        context = get_docker_context_for_node(host)
        env = os.environ.copy()
        env["LOCAL_UID"] = str(os.getuid())
        env["LOCAL_GID"] = str(os.getgid())

        project_name = f"pseti-daqnode-{host.replace('.', '-')}"

        compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "docker-compose.daqnode.yml"
        cmd = [
            "docker", "--context", context,
            "compose", "-p", project_name, "-f", str(compose_file),
            "build"
        ]
        run_cmd(host, cmd, env=env)

        from control.utils.util import is_local
        from control.utils.config_file import get_daq_config
        is_headnode = is_local(host, get_daq_config())
        if not is_headnode:
            alloy_compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "alloy" / "docker-compose.alloy.yml"
            alloy_cmd = [
                "docker", "--context", context,
                "compose", "-p", project_name, "-f", str(alloy_compose_file),
                "build"
            ]
            run_cmd(host, alloy_cmd, env=env)


@app.command()
def down(
    nodes: Annotated[str, typer.Argument(help="Comma-separated list of hostnames/IPs, 'headnode', or 'all' (every DAQ node + the head node).")],
    mode: Annotated[str, typer.Option("--mode", help="'docker' or 'bare-metal' deployment strategy.")] = "docker"
) -> None:
    """Tear down the DAQ node gRPC/telemetry stack and/or the head node stack."""
    if mode != "docker":
        console.print("[bold red]Error:[/] down is only supported in docker mode.")
        raise typer.Exit(1)

    daq_targets, include_headnode = resolve_target_nodes(nodes)

    if include_headnode:
        env = get_headnode_compose_env()
        if env is not None:
            compose_file = PanoPaths.base_dir() / "deploy" / "docker-compose.headnode.yml"
            cmd = ["docker", "compose", "-p", "pseti-headnode", "-f", str(compose_file), "down"]
            run_cmd("headnode", cmd, env=env)

    for host in daq_targets:
        context = get_docker_context_for_node(host)
        project_name = f"pseti-daqnode-{host.replace('.', '-')}"
        
        compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "docker-compose.daqnode.yml"
        cmd = [
            "docker", "--context", context,
            "compose", "-p", project_name, "-f", str(compose_file),
            "down"
        ]
        run_cmd(host, cmd)

        from control.utils.util import is_local
        from control.utils.config_file import get_daq_config
        is_headnode = is_local(host, get_daq_config())
        if not is_headnode:
            alloy_compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "alloy" / "docker-compose.alloy.yml"
            alloy_cmd = [
                "docker", "--context", context,
                "compose", "-p", project_name, "-f", str(alloy_compose_file),
                "down"
            ]
            run_cmd(host, alloy_cmd)

@app.command()
def attach(
    node: Annotated[str, typer.Argument(help="Hostname/IP of DAQ node or 'headnode'.")],
    service: Annotated[str, typer.Argument(help="Service to attach to (e.g. daqnode-server, alloy).")] = "daqnode-server"
) -> None:
    """Tail logs for a specific service on a DAQ node or head node."""
    if node == "headnode":
        env = get_headnode_compose_env()
        if env is not None:
            compose_file = PanoPaths.base_dir() / "deploy" / "docker-compose.headnode.yml"
            cmd = ["docker", "compose", "-p", "pseti-headnode", "-f", str(compose_file), "logs", "-f", service]
            subprocess.run(cmd, env=env)
    else:
        context = get_docker_context_for_node(node)
        project_name = f"pseti-daqnode-{node.replace('.', '-')}"
        
        # Decide which compose file has the service
        if service == "alloy":
            compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "alloy" / "docker-compose.alloy.yml"
        else:
            compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "docker-compose.daqnode.yml"
            
        cmd = [
            "docker", "--context", context,
            "compose", "-p", project_name, "-f", str(compose_file),
            "logs", "-f", service
        ]
        subprocess.run(cmd)

@app.command()
def status(
    nodes: Annotated[str, typer.Argument(help="Comma-separated list of hostnames/IPs, 'headnode', or 'all' (every DAQ node + the head node).")],
    mode: Annotated[str, typer.Option("--mode", help="'docker' or 'bare-metal' deployment strategy.")] = "docker"
) -> None:
    """Check the status of the DAQ node services and/or the head node stack."""
    daq_targets, include_headnode = resolve_target_nodes(nodes)

    if include_headnode:
        status_headnode(mode)

    for host in daq_targets:
        if mode == "docker":
            context = get_docker_context_for_node(host)
            project_name = f"pseti-daqnode-{host.replace('.', '-')}"
            
            compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "docker-compose.daqnode.yml"
            cmd = ["docker", "--context", context, "compose", "-p", project_name, "-f", str(compose_file), "ps"]
            run_cmd(host, cmd)

            from control.utils.util import is_local
            from control.utils.config_file import get_daq_config
            is_headnode = is_local(host, get_daq_config())
            if not is_headnode:
                alloy_compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "alloy" / "docker-compose.alloy.yml"
                alloy_cmd = ["docker", "--context", context, "compose", "-p", project_name, "-f", str(alloy_compose_file), "ps"]
                run_cmd(host, alloy_cmd)
        else:
            cmd = ["ssh", host, "systemctl is-active panoseti_grpc panoseti_alloy"]
            run_cmd(host, cmd)

if __name__ == "__main__":
    app()
