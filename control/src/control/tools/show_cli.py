from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
from typing import Annotated, Any

import numpy as np
import typer
from grpc.aio import AioRpcError
from panoseti_grpc.daq_control.client import AsyncDaqControlClient
from panoseti_grpc.daq_data.client import AioDaqDataClient
from panoseti_grpc.util.cli import display_tree_callback
from rich import print
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from control.utils import config_file, util
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
    scale = ' .:-=+*#%@▒▓█'
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
    init: bool = False,
    legend: bool = False,
) -> None:
    """Async generator-driven science data stream."""
    daq_config = config_file.get_daq_config()
    network_config = config_file.get_network_config()
    util.attach_daq_config(daq_config, network_config)
    
    # dict conversion for client
    daq_cfg_dict = daq_config if isinstance(daq_config, dict) else daq_config.model_dump()
    net_cfg_dict = network_config if isinstance(network_config, dict) else network_config.model_dump()

    # module_id -> type -> {quabo_id or 'full': pano_image}
    latest_images: dict[int, dict[str, dict[Any, Any]]] = {}

    async with AioDaqDataClient(daq_cfg_dict, net_cfg_dict) as client:
        hosts = await client.get_valid_daq_hosts()
        if not hosts:
            print("[red]No valid DAQ hosts found. Ensure daq_data services are running.[/red]")
            return

        # Check status and optionally initialize
        status_ok = True
        for h_str in hosts:
            s = await client.status(h_str)
            if not s or not s.hp_io_initialized:
                if init:
                    # Find matching DaqNode to get data_dir and check hashpipe liveness
                    matching_node = None
                    for node in daq_config.daq_nodes:
                        endpoint_h, endpoint_p = util.daq_grpc_endpoint(node, daq_config)
                        if f"{endpoint_h}:{endpoint_p}" == h_str:
                            matching_node = node
                            break
                    
                    if matching_node:
                        h, p = util.daq_grpc_endpoint(matching_node, daq_config)
                        async with AsyncDaqControlClient(host=h, port=p) as control_client:
                            try:
                                ok, daq_status = await control_client.StatusDaq({
                                    'data_dir': matching_node.data_dir,
                                    'check_hashpipe_running': True
                                }, timeout=5.0)
                                
                                if ok and daq_status.get('hashpipe_running'):
                                    hp_io_cfg = {
                                        "update_interval_seconds": 0.1,
                                        "force": True,
                                        "simulate_daq": False,
                                        "module_ids": []
                                    }
                                    success = await client.init_hp_io(h_str, hp_io_cfg)
                                    if success:
                                        print(f"[green]Successfully initialized DaqData hp_io on {h_str}.[/green]")
                                        continue
                                    else:
                                        print(f"[red]Failed to initialize DaqData hp_io on {h_str}.[/red]")
                                else:
                                    print(f"[yellow]Warning: Hashpipe is NOT running on {h_str}. Cannot initialize DaqData.[/yellow]")
                            except Exception as e:
                                print(f"[red]Error checking hashpipe status on {h_str}: {e}[/red]")
                
                print(f"[yellow]Warning: DaqData hp_io is not initialized on {h_str}.[/yellow]")
                status_ok = False
        
        if not status_ok:
            print("[dim]Tip: Run 'pseti show sci --init' or 'pseti start --init-snapshot' to initialize the data streaming service.[/dim]")


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
                    img_type = pano_image.get("type", "UNDEFINED")
                    shape = pano_image.get("shape")
                    
                    if mid not in latest_images:
                        latest_images[mid] = {}
                    if img_type not in latest_images[mid]:
                        latest_images[mid][img_type] = {}
                    
                    if shape == [32, 32]:
                        latest_images[mid][img_type]['full'] = pano_image
                    elif shape == [16, 16]:
                        # Identify quabo from header
                        qid = 0
                        header = pano_image.get('header', {})
                        for i in range(4):
                            if f'quabo_{i}' in header:
                                qid = i
                                break
                        latest_images[mid][img_type][qid] = pano_image
                    
                    # Build display
                    sorted_mids = sorted(latest_images.keys())
                    display_group = []
                    
                    for m in sorted_mids:
                        module_data = latest_images[m]
                        for t in sorted(module_data.keys()):
                            data = module_data[t]
                            img_to_render = None
                            shape_to_render = None
                            bpp = 2
                            header = {}
                            frame_num = 0
                            
                            if 'full' in data:
                                img_dict = data['full']
                                img_to_render = img_dict['image_array']
                                shape_to_render = img_dict['shape']
                                bpp = img_dict.get('bytes_per_pixel', 2)
                                header = img_dict.get('header', {})
                                frame_num = img_dict.get('frame_number', 0)
                            else:
                                qids = [q for q in data if isinstance(q, int)]
                                if not qids:
                                    continue
                                
                                # Use info from the most recent quadrant we've seen
                                latest_q = data[max(qids)]
                                bpp = latest_q.get('bytes_per_pixel', 2)
                                header = latest_q.get('header', {})
                                frame_num = latest_q.get('frame_number', 0)
                                
                                # Assembled view is always 32x32
                                merged = np.zeros((32, 32), dtype=np.uint16)
                                for qid in qids:
                                    q_img = data[qid]
                                    q_arr = q_img['image_array']
                                    # PANOSETI quadrant layout: 0 1 / 2 3
                                    row_off = (qid // 2) * 16
                                    col_off = (qid % 2) * 16
                                    merged[row_off:row_off+16, col_off:col_off+16] = q_arr
                                img_to_render = merged
                                shape_to_render = [32, 32]
                            
                            if img_to_render is not None:
                                # Calculate stats for legend
                                flat = img_to_render.flatten()
                                v_min = float(np.min(flat))
                                v_max = float(np.max(flat))
                                
                                img_text = render_image_text(
                                    img_to_render, 
                                    shape_to_render, 
                                    bpp,
                                    min_val=v_min,
                                    max_val=v_max
                                )
                                ts = header.get("pandas_unix_timestamp", "N/A")
                                info = Text(f"Module {m} | {t} | Frame {frame_num}\n{ts}", style="dim")
                                
                                content: list[Any] = [info, img_text]
                                
                                if legend:
                                    v_25 = float(np.percentile(flat, 25))
                                    v_50 = float(np.percentile(flat, 50))
                                    v_75 = float(np.percentile(flat, 75))
                                    
                                    scale = ' .:-=+*#%@▒▓█'
                                    legend_text = Text("\nLegend (ADC):\n", style="bold")
                                    legend_text.append(f"Min: {v_min:.0f} | 25%: {v_25:.0f} | 50%: {v_50:.0f} | 75%: {v_75:.0f} | Max: {v_max:.0f}\n", style="dim")
                                    
                                    # Show a small sample of the scale mapping
                                    mapping = "Scale: "
                                    for i, char in enumerate(scale):
                                        val = v_min + (i / (len(scale)-1)) * (v_max - v_min)
                                        mapping += f"{char}:{val:.0f} "
                                    legend_text.append(mapping, style="dim")
                                    content.append(legend_text)
                                
                                panel = Panel(
                                    Group(*content),
                                    title=f"Module {m} - {t}",
                                    border_style="green" if t == "MOVIE" else "magenta",
                                    padding=(0, 1)
                                )
                                display_group.append(panel)
                    
                    if not display_group:
                        live.update(Text("Waiting for data..."))
                        continue

                    # Tile the panels efficiently
                    live.update(Columns(display_group, expand=True, equal=True))


@app.command(name="sci")
def show_sci(
    interval: Annotated[float, typer.Option("--interval", "-i", help="Update interval in seconds.")] = 1.0,
    module_ids: Annotated[list[int], typer.Option("--module", "-m", help="Whitelist of module IDs to display.")] = [],
    movie: Annotated[bool, typer.Option("--movie/--no-movie", help="Stream movie-mode images.")] = True,
    ph: Annotated[bool, typer.Option("--ph/--no-ph", help="Stream pulse-height images.")] = True,
    init: Annotated[bool, typer.Option("--init", help="Attempt to initialize gRPC servers on all daq nodes.")] = False,
    legend: Annotated[bool, typer.Option("--legend", help="Display ADC quantiles and symbol mapping legend.")] = False,
) -> None:
    """
    Display a live-updating text view of the science data stream.
    """
    with contextlib.suppress(KeyboardInterrupt, AioRpcError):
        asyncio.run(stream_sci_data(interval, module_ids, movie, ph, init, legend))


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
