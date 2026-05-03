from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
from typing import Annotated, Any

import numpy as np
import typer
from panoseti_grpc.daq_data.client import AioDaqDataClient
from panoseti_grpc.util.cli import display_tree_callback
from rich import print
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from control.utils import config_file
from control.utils.paths import PanoPaths

app = typer.Typer(help="Inspect and visualize PSETI system state.", no_args_is_help=True)


@app.callback()
def show_callback(
    ctx: typer.Context,
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree for system inspection.", callback=display_tree_callback)] = False
) -> None:
    """Inspect and visualize system state."""
    pass


@app.command(name="paths")
def show_paths(
    tree_opt: Annotated[bool, typer.Option("--tree", "-t", help="Display paths as a file tree.")] = False
) -> None:
    """
    Display the current resolved paths for all key directories.
    
    To override any of these paths, set the corresponding environment variable.
    
    Examples:
      $ pseti show paths
      $ pseti show paths --tree
      $ export PSETI_CONFIG=/tmp/custom_configs
    """
    console = Console()
    
    paths = [
        ("Repository Root", PanoPaths.software_root_dir(), "PSETI_ROOT"),
        ("Control Package", PanoPaths.base_dir(), "PSETI_CONTROL"),
        ("Configs", PanoPaths.config_dir(), "PSETI_CONFIG"),
        ("Transient (tmp)", PanoPaths.tmp_dir(), "PSETI_TMP"),
        ("Quabos Metadata", PanoPaths.quabos_dir(), "PSETI_QUABOS"),
        ("Logs", PanoPaths.logs_dir(), "PSETI_LOGS"),
        ("Firmware", PanoPaths.firmware_dir(), "PSETI_FIRMWARE"),
        ("White Rabbit", PanoPaths.wr_dir(), "PSETI_WR"),
        ("DAQ Scripts", PanoPaths.daq_scripts_dir(), "PSETI_DAQ_SCRIPTS"),
        ("State Root", PanoPaths.state_dir(), "PSETI_STATE"),
        ("Locks", PanoPaths.locks_dir(), "PSETI_LOCKS_DIR"),
        ("Run State", PanoPaths.runs_dir(), "PSETI_RUNS_DIR"),
        ("Transfer Queue", PanoPaths.transfer_queue_dir(), "PSETI_TQ_DIR"),
        ("Transfer Manifests", PanoPaths.transfer_manifests_dir(), "PSETI_TM_DIR"),
        ("Calibration", PanoPaths.calibration_dir(), "PSETI_CALIB_DIR"),
    ]

    if tree_opt:
        # Build a tree representation
        # We'll use the Repository Root as the base if possible
        root_path = PanoPaths.software_root_dir()
        tree = Tree(f"[bold blue]{root_path}[/bold blue] (PSETI_ROOT)")
        
        # Dictionary to keep track of tree nodes
        nodes: dict[pathlib.Path, Tree] = {root_path: tree}

        # Sort paths by depth to build tree incrementally
        sorted_paths = sorted(paths, key=lambda x: len(x[1].parts))

        for name, path, var in sorted_paths:
            if path == root_path:
                continue
            
            # Find the best parent in our nodes
            parent_path = path.parent
            while parent_path not in nodes and parent_path != parent_path.parent:
                parent_path = parent_path.parent
            
            parent_node = nodes.get(parent_path, tree)
            
            # Calculate relative path from parent
            try:
                rel_path = path.relative_to(parent_path)
            except ValueError:
                rel_path = path # Absolute if not relative
            
            label = f"[green]{rel_path}[/green] [dim]({name})[/dim]"
            if os.environ.get(var):
                label += f" [bold magenta]OVERRIDDEN by {var}[/bold magenta]"
            
            node = parent_node.add(label)
            nodes[path] = node
        
        console.print(tree)
    else:
        table = Table(title="PSETI Path Mapping")
        table.add_column("Directory", style="cyan", no_wrap=True)
        table.add_column("Resolved Path", style="green", overflow="fold")
        table.add_column("Override Variable", style="magenta", no_wrap=True)
        table.add_column("Source", style="blue", no_wrap=True)

        for name, path, var in paths:
            source = f"[bold]{var}[/bold]" if os.environ.get(var) else "Default"
            table.add_row(name, str(path), var, source)

        console.print(table)
    
    console.print("\n[dim]Tip: Overriding PSETI_ROOT or PSETI_CONTROL will shift the default locations of all sub-directories.[/dim]")


def render_image_text(img_array: Any, shape: list[int], bpp: int, min_val: float = 0, max_val: float = 0) -> Text:
    """Renders a PanoImage as a rich Text object using density scaling."""
    scale = ' .,-+=#@'
    img_size_y, img_size_x = shape
    text = Text()
    
    # Simple auto-scale if min/max not provided
    if min_val == 0 and max_val == 0:
        flat = img_array.flatten()
        if len(flat) > 0:
            min_val = float(np.min(flat)) if hasattr(flat, "min") else float(min(flat))
            max_val = float(np.max(flat)) if hasattr(flat, "max") else float(max(flat))

    for row in range(img_size_y):
        line = ""
        for col in range(img_size_x):
            val = img_array[row][col] if hasattr(img_array, "shape") else img_array[row * img_size_x + col]
            if max_val != min_val:
                y = (val - min_val) / (max_val - min_val)
                y = max(0.0, min(1.0, float(y)))
                idx = int(y * (len(scale) - 1))
            else:
                idx = val // 8192 if bpp == 2 else val // 32
                idx = max(0, min(len(scale) - 1, int(idx)))
                if val > 0 and idx == 0:
                    idx = 1
            line += scale[idx] + " "
        text.append(line + "\n")
    return text


async def stream_sci_data(
    interval: float,
    module_ids: list[int],
    movie: bool,
    ph: bool,
) -> None:
    """Async generator-driven science data stream."""
    daq_config = config_file.get_daq_config()
    network_config = config_file.get_network_config()
    
    # dict conversion for client
    daq_cfg_dict = daq_config if isinstance(daq_config, dict) else daq_config.model_dump()
    net_cfg_dict = network_config if isinstance(network_config, dict) else network_config.model_dump()

    latest_images: dict[int, dict[str, Any]] = {}

    async with AioDaqDataClient(daq_cfg_dict, net_cfg_dict) as client:
        hosts = await client.get_valid_daq_hosts()
        if not hosts:
            print("[red]No valid DAQ hosts found. Ensure daq_data services are running.[/red]")
            return

        # Check status
        status_ok = True
        for h in hosts:
            s = await client.status(h)
            if not s or not s.hp_io_initialized:
                print(f"[yellow]Warning: DaqData hp_io is not initialized on {h}.[/yellow]")
                status_ok = False
        
        if not status_ok:
            print("[dim]Tip: Run 'pseti start --init-hp-io' to initialize the data service.[/dim]")

        stream = await client.stream_images(
            hosts=hosts,
            stream_movie_data=movie,
            stream_pulse_height_data=ph,
            update_interval_seconds=interval,
            module_ids=tuple(module_ids),
            parse_pano_images=True
        )

        console = Console()
        with Live(Text("Waiting for data..."), console=console, refresh_per_second=4) as live:
            async for pano_image in stream:
                if isinstance(pano_image, dict):
                    mid = pano_image.get("module_id", 0)
                    latest_images[mid] = pano_image
                    
                    # Build display
                    # display_group = []
                    sorted_mids = sorted(latest_images.keys())
                    
                    # Create a layout for modules (2 per row if possible)
                    table = Table.grid(expand=True)
                    table.add_column()
                    table.add_column()
                    
                    current_row = []
                    for m in sorted_mids:
                        img = latest_images[m]
                        img_text = render_image_text(
                            img["image_array"], 
                            img["shape"], 
                            img["bytes_per_pixel"]
                        )
                        header = img.get("header", {})
                        ts = header.get("pandas_unix_timestamp", "N/A")
                        info = Text(f"Module {m} | {img['type']} | Frame {img['frame_number']}\n{ts}", style="dim")
                        
                        panel = Panel(
                            Group(info, img_text),
                            title=f"Module {m}",
                            border_style="green"
                        )
                        current_row.append(panel)
                        if len(current_row) == 2:
                            table.add_row(*current_row)
                            current_row = []
                    if current_row:
                        table.add_row(*current_row)
                    
                    live.update(table)


@app.command(name="sci")
def show_sci(
    interval: Annotated[float, typer.Option("--interval", "-i", help="Update interval in seconds.")] = 1.0,
    module_ids: Annotated[list[int], typer.Option("--module", "-m", help="Whitelist of module IDs to display.")] = [],
    movie: Annotated[bool, typer.Option("--movie/--no-movie", help="Stream movie-mode images.")] = True,
    ph: Annotated[bool, typer.Option("--ph/--no-ph", help="Stream pulse-height images.")] = False,
) -> None:
    """
    Display a live-updating text view of the science data stream.
    """
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(stream_sci_data(interval, module_ids, movie, ph))


@app.command(name="commands")
def show_commands(
    ctx: typer.Context,
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree.", callback=display_tree_callback)] = True
) -> None:
    """
    Display a tree-like view of all available PSETI commands and subcommands.
    """
    # This command is now just an alias for -t at this level
    pass


if __name__ == "__main__":
    app()
