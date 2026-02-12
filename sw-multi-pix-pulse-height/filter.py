#! /usr/bin/env python3
"""
panoseti_filter.py

Generic PFF Filter.
Usage:
    python panoseti_filter.py input_dir output_dir --kernel neighbor --save
"""

import argparse
import logging
import time
import psutil
from pathlib import Path
from typing import Dict, Any
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

# Rich Imports
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeRemainingColumn, FileSizeColumn
)
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel


logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

# Setup Console
console = Console()
logging.basicConfig(
    level="WARNING",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True)]
)
logger = logging.getLogger("PanoFilter")
logger.setLevel(logging.INFO)

# Interfaces
try:
    from panoseti_interface import PFFSequence, PanosetiRun
    from jax_filters import KERNELS
except ImportError as e:
    logger.exception("CRITICAL: Missing modules. Ensure panoseti_interface.py and jax_filters.py are present.")
    exit(1)



def parse_params(param_str: str) -> Dict[str, Any]:
    """Parses 'k=v,k2=v2' string into dict."""
    params = {}
    if not param_str: return params
    for pair in param_str.split(','):
        k, v = pair.split('=')
        try:
            if '.' in v:
                val = float(v)
            else:
                val = int(v)
        except ValueError:
            val = v
        params[k] = val
    return params


def get_system_metrics():
    """Returns formatted string of CPU/RAM usage."""
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent()
    return f"CPU: {cpu:>4.1f}% | RAM: {mem.percent:>4.1f}% ({mem.used / 1e9:.1f} GB)"


def run_filter_job(
        run_dir: Path,
        output_dir: Path,
        kernel_name: str,
        params: Dict[str, Any],
        product_filter: str = "all",
        save_output: bool = False,
        batch_size: int = 5000
):
    # 1. Setup
    if not run_dir.exists():
        logger.error(f"Run dir not found: {run_dir}")
        return

    kernel_func = KERNELS.get(kernel_name)
    if not kernel_func:
        logger.error(f"Kernel '{kernel_name}' not found.")
        return

    # JAX Setup
    def batched_kernel_wrapper(batch, **p):
        bound_func = partial(kernel_func, **p)
        return jax.vmap(bound_func)(batch)

    # 2. Scan Products
    run = PanosetiRun(run_dir)
    if save_output:
        output_dir.mkdir(parents=True, exist_ok=True)

    all_prods = run.list_products()
    targets = [p for p in all_prods if product_filter == "all" or product_filter in p]

    # Pre-calculate totals
    total_files = len(targets)

    # 3. Layout Construction
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )

    stats_table = Table(expand=True)
    stats_table.add_column("Product", style="cyan")
    stats_table.add_column("Frames", justify="right")
    stats_table.add_column("Kept", justify="right", style="green")
    stats_table.add_column("Ratio", justify="right")
    stats_table.add_column("Speed", justify="right")

    job_info = f"[bold]Job:[/][cyan]{kernel_name}[/] {params} | [bold]Mode:[/]{'SAVE' if save_output else 'DRY RUN'}"
    layout["header"].update(Panel(job_info, title="PANOSETI Filter Job"))
    layout["main"].update(stats_table)

    # 4. Processing Loop
    with Live(layout, refresh_per_second=4, console=console) as live:

        # Overall Progress
        for i, prod_name in enumerate(targets):
            seq = run.get_product(prod_name)
            layout["footer"].update(Panel(f"Processing {i + 1}/{total_files}: {prod_name} | {get_system_metrics()}"))

            try:
                res = _process_sequence(
                    seq, output_dir, batched_kernel_wrapper, params, batch_size, save_output
                )

                speed_str = f"{res['fps']:,.0f} fps"
                stats_table.add_row(
                    seq.name,
                    f"{res['total']:,}",
                    f"{res['kept']:,}",
                    f"{res['ratio']:.2f}%",
                    speed_str
                )
            except Exception as e:
                logger.exception(f"Failed {prod_name}")
                stats_table.add_row(seq.name, "ERROR", "-", "-", "-")

    console.print(f"[bold green]Job Complete.[/]")


def _process_sequence(seq, out_dir, kernel_fn, params, batch_size, save_output):
    """
    Process a single sequence.
    """
    # Setup Output
    out_path = None
    f_out = None
    if save_output:
        param_tag = ".".join([f"{k}_{v}" for k, v in params.items()])
        src_stem = seq.file_paths[0].name.split('.seqno')[0]
        out_name = f"{src_stem}.{param_tag}.seqno_0.pff"
        out_path = out_dir / out_name
        f_out = open(out_path, 'wb')

    start_t = time.perf_counter()
    total_kept = 0
    total_frames = len(seq)

    try:
        # Loop through data in batches
        for chunk_start in range(0, total_frames, batch_size):
            # Optimized fetch (mmap strided copy)
            img_batch = seq.get_image_array(chunk_start, batch_size)
            if len(img_batch) == 0: break

            # JAX Execution (Compute)
            # We assume img_batch fits in VRAM/RAM. 5000 * 2KB = ~10MB. Safe.
            # Convert to JAX array (moves to GPU if available)
            jax_batch = jnp.array(img_batch)
            batch_results = kernel_fn(jax_batch, **params)

            # Extract Keep Mask
            # Result is tuple: (trigger_mask, supported_mask, ..., keep_bools)
            # We defined last element as the boolean keep decision
            if isinstance(batch_results, (tuple, list)):
                keep_mask = np.array(batch_results[-1])
            else:
                keep_mask = np.array(batch_results)

            keep_indices = np.where(keep_mask)[0]
            count_kept = len(keep_indices)
            total_kept += count_kept

            # Write I/O (Optional)
            if save_output and count_kept > 0:
                # We need to fetch raw bytes (Header + Img) for kept frames
                # Optimized: We can read directly from mmap without parsing again
                # But we need to handle the specific offsets.
                # Since get_image_array logic is complex, let's just use the robust single fetch
                # or a new bulk raw fetch. For safety, we iterate kept indices.

                # NOTE: Optimization opportunity:
                # If we kept a lot, raw file copying is faster than seeking.
                # But typically we keep < 1% of data. Seeking is fine.
                for local_idx in keep_indices:
                    global_idx = chunk_start + local_idx

                    # Manual fetch to avoid overhead
                    # Locate file
                    # (This repeats logic, but is safe. Ideally move to interface)
                    _head, _img = seq.get_frame(global_idx)
                    # Note: We need RAW bytes (Header + Payload), not parsed.
                    # Re-implement raw read here or add method to Interface.
                    # For now, let's trust the Interface's file path logic
                    # but implement a raw read here for speed.

                    # Find file logic simplified:
                    # (Assuming seq stores logic to find file index)
                    # We will use a helper we add below or just use standard read for correctness

                    # *SLOW PATH*: seq.get_frame parses JSON. We want raw bytes.
                    # Use internal private access for speed or add public method.
                    # Let's use the file_paths directly based on logic

                    # Calculate offsets
                    f_idx = 0
                    l_idx = global_idx
                    for i, limit in enumerate(seq._cumulative_frames):
                        if global_idx < limit:
                            f_idx = i
                            l_idx = global_idx - (seq._cumulative_frames[i - 1] if i > 0 else 0)
                            break

                    mm = seq._get_mmap(f_idx)
                    off = l_idx * seq.frame_config.frame_size
                    end = off + seq.frame_config.frame_size
                    f_out.write(mm[off:end])

    finally:
        if f_out: f_out.close()

    dt = time.perf_counter() - start_t
    if dt == 0: dt = 0.001

    return {
        'total': total_frames,
        'kept': total_kept,
        'ratio': (total_kept / total_frames) * 100 if total_frames else 0,
        'fps': total_frames / dt
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PANOSETI Generic Filter")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--kernel", type=str, default="neighbor")
    parser.add_argument("--params", type=str, default="thresh=150,n_min=3")
    parser.add_argument("--product", type=str, default="all")
    parser.add_argument("--save", action="store_true", help="Enable writing output to disk (Default: Dry Run)")

    args = parser.parse_args()

    run_filter_job(
        args.run_dir,
        args.output_dir,
        args.kernel,
        parse_params(args.params),
        args.product,
        args.save
    )