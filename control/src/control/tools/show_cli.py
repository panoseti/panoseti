from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
from typing import Annotated, Any

import numpy as np
import typer
from grpc.aio import AioRpcError
from matplotlib import colormaps
from panoseti_grpc.daq_data.client import AioDaqDataClient
from panoseti_grpc.util.cli import display_tree_callback
from rich import print
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from control.utils.paths import PanoPaths

app = typer.Typer(help="Inspect and visualize PSETI system state.", no_args_is_help=True)


@app.callback()
def show_callback(
    ctx: typer.Context,
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree for system inspection.", callback=display_tree_callback)] = False
) -> None:
    """Inspect and visualize system state (sci data)."""
    pass


def show_paths(
    tree_opt: Annotated[bool, typer.Option("--tree", "-t", help="Display paths as a file tree.")] = False
) -> None:
    """
    Display the current resolved paths for all key directories.
    
    To override any of these paths, set the corresponding environment variable.
    
    Examples:
      $ pseti paths
      $ pseti paths --tree
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


def get_color(val: float, cmap_name: str | None) -> str:
    """Get RGB string from matplotlib colormap or grayscale."""
    if not cmap_name or cmap_name.lower() == "none":
        v = int(val * 255)
        return f"rgb({v},{v},{v})"
    
    try:
        cmap = colormaps.get_cmap(cmap_name)
    except ValueError:
        # Fallback to grayscale if cmap not found
        v = int(val * 255)
        return f"rgb({v},{v},{v})"
    
    rgba = cmap(val)
    r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    return f"rgb({r},{g},{b})"


def render_image_text(
    img_array: Any, 
    shape: list[int], 
    bpp: int, 
    min_val: float = 0, 
    max_val: float = 0,
    color_palette: str | None = "viridis",
    compact: bool = True
) -> Text:
    """Renders a PanoImage as a rich Text object using density scaling or half-blocks."""
    img_size_y, img_size_x = shape
    text = Text()
    
    # Simple auto-scale if min/max not provided
    if min_val == 0 and max_val == 0:
        flat = img_array.flatten()
        if len(flat) > 0:
            min_val = float(np.min(flat)) if hasattr(flat, "min") else float(min(flat))
            max_val = float(np.max(flat)) if hasattr(flat, "max") else float(max(flat))

    if compact:
        # High-resolution half-block mode (2 pixels per character cell)
        for row in range(0, img_size_y, 2):
            for col in range(img_size_x):
                # Handle potential odd row count (though PanoSETI is always 16 or 32)
                v1 = img_array[row][col] if hasattr(img_array, "shape") else img_array[row * img_size_x + col]
                y1 = (v1 - min_val) / (max_val - min_val) if max_val != min_val else 0
                y1 = max(0.0, min(1.0, float(y1)))
                
                if row + 1 < img_size_y:
                    v2 = img_array[row+1][col] if hasattr(img_array, "shape") else img_array[(row+1) * img_size_x + col]
                    y2 = (v2 - min_val) / (max_val - min_val) if max_val != min_val else 0
                    y2 = max(0.0, min(1.0, float(y2)))
                else:
                    y2 = 0
                
                color_top = get_color(y1, color_palette)
                color_bottom = get_color(y2, color_palette)
                text.append("▀", style=f"{color_top} on {color_bottom}")
            text.append("\n")
    else:
        # Density scale mode (1 pixel per character cell + space for aspect ratio)
        scale = ' .:-=+*#%@▒▓█'
        for row in range(img_size_y):
            for col in range(img_size_x):
                val = img_array[row][col] if hasattr(img_array, "shape") else img_array[row * img_size_x + col]
                if max_val != min_val:
                    y = (val - min_val) / (max_val - min_val)
                    y = max(0.0, min(1.0, float(y)))
                    idx = int(y * (len(scale) - 1))
                else:
                    y = (val // 8192 if bpp == 2 else val // 32) / (len(scale)-1)
                    y = max(0.0, min(1.0, float(y)))
                    idx = int(y * (len(scale) - 1))
                
                char = scale[idx]
                color = get_color(y, color_palette)
                text.append(char + " ", style=color)
            text.append("\n")
    return text


async def stream_sci_data(
    interval: float,
    module_ids: list[int],
    movie: bool,
    ph: bool,
    init: bool = False,
    init_sim: bool = False,
    legend_local: bool = False,
    legend_global: bool = False,
    color: str | None = "viridis",
    compact: bool = True,
) -> None:
    """Async science data stream with decoupled rendering to prevent flickering."""
    gateway_host = os.getenv("DAQ_DATA_GATEWAY_HOST", "localhost")
    gateway_port = int(os.getenv("DAQ_DATA_GATEWAY_PORT", "50051"))

    # Shared state between ingestion and rendering
    latest_images: dict[int, dict[str, dict[Any, Any]]] = {}
    dirty = False

    async with AioDaqDataClient(gateway_host, gateway_port) as client:
        # Handle initialization if requested
        if init_sim:
            await client.init_sim()
            print("[green]Initialized simulation stream.[/green]")
        elif init:
            hp_io_cfg = {"update_interval_seconds": 0.1, "force": True}
            await client.init_hp_io(hp_io_cfg)
            print("[green]Initialized science stream.[/green]")

        stream = client.stream_images(
            stream_movie_data=movie,
            stream_pulse_height_data=ph,
            update_interval_seconds=interval,
            module_ids=tuple(module_ids),
            parse_pano_images=True,
        )

        console = Console()

        async def ingestion_task() -> None:
            nonlocal dirty
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
                        qid = 0
                        header = pano_image.get('header', {})
                        for i in range(4):
                            if f'quabo_{i}' in header:
                                qid = i
                                break
                        latest_images[mid][img_type][qid] = pano_image
                    dirty = True

        # Start ingestion in the background
        ingest_fut = asyncio.create_task(ingestion_task())

        try:
            with Live(Text("Waiting for data..."), console=console, refresh_per_second=2) as live:
                while not ingest_fut.done():
                    if not dirty:
                        await asyncio.sleep(0.05)
                        continue
                    
                    dirty = False
                    
                    # Compute global stats if requested
                    g_min, g_max = 0.0, 0.0
                    g_flat = np.array([])
                    if legend_global:
                        all_arrays = []
                        for m_id in latest_images:
                            for t_id in latest_images[m_id]:
                                d = latest_images[m_id][t_id]
                                if 'full' in d:
                                    all_arrays.append(d['full']['image_array'].flatten())
                                else:
                                    for q in d:
                                        if isinstance(q, int):
                                            all_arrays.append(d[q]['image_array'].flatten())
                        if all_arrays:
                            g_flat = np.concatenate(all_arrays)
                            g_min = float(np.min(g_flat))
                            g_max = float(np.max(g_flat))

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
                                latest_q = data[max(qids)]
                                bpp = latest_q.get('bytes_per_pixel', 2)
                                header = latest_q.get('header', {})
                                frame_num = latest_q.get('frame_number', 0)
                                merged = np.zeros((32, 32), dtype=np.uint16)
                                for qid in qids:
                                    q_arr = data[qid]['image_array']
                                    row_off, col_off = (qid // 2) * 16, (qid % 2) * 16
                                    merged[row_off:row_off+16, col_off:col_off+16] = q_arr
                                img_to_render, shape_to_render = merged, [32, 32]
                            
                            if img_to_render is not None:
                                flat = img_to_render.flatten()
                                l_min, l_max = float(np.min(flat)), float(np.max(flat))
                                use_min = g_min if legend_global else l_min
                                use_max = g_max if legend_global else l_max
                                
                                img_text = render_image_text(
                                    img_to_render, shape_to_render, bpp,
                                    min_val=use_min, max_val=use_max,
                                    color_palette=color, compact=compact
                                )
                                ts = header.get("pandas_unix_timestamp", "N/A")
                                info = Text(f"M{m}|{t}|F{frame_num} {ts}", style="dim")
                                content: list[Any] = [info, img_text]
                                
                                if legend_local or legend_global:
                                    stats_flat = g_flat if legend_global else flat
                                    
                                    quants = [0, 0.25, 0.5, 0.75, 1.0]
                                    labels = ["Min", "25%", "50%", "75%", "Max"]
                                    vals = np.percentile(stats_flat, [q*100 for q in quants])
                                    
                                    legend_text = Text(f"{'Global' if legend_global else 'Local'} ADC: ", style="bold")
                                    for lbl, val, q in zip(labels, vals, quants, strict=True):
                                        c_str = get_color(q, color)
                                        sym = "█" 
                                        legend_text.append(f"{lbl}:", style="dim")
                                        legend_text.append(f"{val:.0f}", style="bold")
                                        legend_text.append(f"{sym} ", style=c_str)
                                    content.append(legend_text)
                                
                                panel = Panel(
                                    Group(*content),
                                    title=f"M{m}-{t}",
                                    border_style="green" if t == "MOVIE" else "magenta",
                                    padding=(0, 0)
                                )
                                display_group.append(panel)
                    
                    if display_group:
                        live.update(Columns(display_group, expand=True, equal=True))
                    await asyncio.sleep(0.1) # UI Refresh interval
        finally:
            ingest_fut.cancel()

@app.command(name="sci")
def show_sci(
    interval: Annotated[float, typer.Option("--interval", "-i", help="Update interval in seconds.")] = 1.0,
    module_ids: Annotated[list[int] | None, typer.Option("--module", "-m", help="Whitelist of module IDs to display.")] = None,
    movie: Annotated[bool, typer.Option("--movie/--no-movie", help="Stream movie-mode images.")] = True,
    ph: Annotated[bool, typer.Option("--ph/--no-ph", help="Stream pulse-height images.")] = True,
    init: Annotated[bool, typer.Option("--init", help="Attempt to initialize gRPC servers on all daq nodes.")] = False,
    init_sim: Annotated[bool, typer.Option("--init-sim", help="Initialize the server with simulation data streaming.")] = False,
    legend_local: Annotated[bool, typer.Option("--legend-local", help="Display per-figure ADC quantiles.")] = False,
    legend_global: Annotated[bool, typer.Option("--legend-global", help="Display global ADC quantiles.")] = False,
    color: Annotated[str, typer.Option("--color", help="Matplotlib colormap (e.g. viridis, inferno, plasma, magma, hot, bone). Default is viridis.")] = "plasma",
    compact: Annotated[bool, typer.Option("--compact/--no-compact", help="Use high-density half-blocks (default).")] = True,
) -> None:
    """
    Display a live-updating text view of the science data stream.
    """
    module_ids = module_ids or []
    palette = None if color.lower() == "none" else color.lower()
    with contextlib.suppress(KeyboardInterrupt, AioRpcError):
        asyncio.run(stream_sci_data(interval, module_ids, movie, ph, init, init_sim, legend_local, legend_global, palette, compact))


@app.command(name="pff")
def show_pff(
    run_dir: Annotated[pathlib.Path, typer.Argument(help="Path to the .pffd run directory.")],
    details: Annotated[bool, typer.Option("--details", "-d", help="Show individual PFF files.")] = False,
) -> None:
    """
    Explore the structure of a PanoSETI run.
    """
    import sys
    # Ensure pypff/src is in path
    repo_root = PanoPaths.software_root_dir()
    pypff_src = (repo_root / "pypff" / "src").resolve()
    if str(pypff_src) not in sys.path:
        sys.path.insert(0, str(pypff_src))
        
    from pypff import PanosetiRun
    
    if not run_dir.exists():
        print(f"[red]❌ Run directory {run_dir} does not exist.[/red]")
        raise typer.Exit(1)
    
    run = PanosetiRun(run_dir)
    run.show(details=details)


if __name__ == "__main__":
    app()
