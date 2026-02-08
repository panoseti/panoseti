#! /usr/bin/env python3
"""
panoseti_filter.py

Generic PFF Filter.
Usage:
    python panoseti_filter.py input_dir output_dir --kernel neighbor --params 'thresh=300,n_min=3' --type img --bpp 2
"""

import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Callable
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Interfaces
try:
    from panoseti_interface import PFFSequence, PanosetiRun
    from jax_filters import KERNELS
except ImportError:
    print("CRITICAL: Missing modules. Ensure panoseti_interface.py and jax_filters.py are present.")
    exit(1)

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

# Setup Rich Console
console = Console()
logging.basicConfig(
    level="WARNING",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True)]
)
logger = logging.getLogger("PanoFilter")
logger.setLevel(logging.INFO)


def parse_params(param_str: str) -> Dict[str, Any]:
    """Parses 'k=v,k2=v2' string into dict, auto-converting ints/floats."""
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


def run_filter_job(
        run_dir: Path,
        output_dir: Path,
        kernel_name: str,
        params: Dict[str, Any],
        product_filter: str = "all",
        type_filter: str = "any",
        bpp_filter: int = 0,
        batch_size: int = 2000
):
    # 1. Setup
    if not run_dir.exists():
        logger.error(f"Run dir not found: {run_dir}")
        return

    kernel_func = KERNELS.get(kernel_name)
    if not kernel_func:
        logger.error(f"Kernel '{kernel_name}' not found. Available: {list(KERNELS.keys())}")
        return

    # JAX Setup
    def batched_kernel_wrapper(batch, **p):
        bound_func = partial(kernel_func, **p)
        return jax.vmap(bound_func)(batch)

    # 2. Scan Products
    run = PanosetiRun(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_prods = run.list_products()

    # Filter Logic
    targets = []
    for pname in all_prods:
        # String match filter
        if product_filter != "all" and product_filter not in pname:
            continue

        # Parse metadata from name for type/bpp checks
        # Name format: dp_ph1024.bpp_2.module_253...
        is_img = "img" in pname
        is_ph = "ph" in pname

        # Type Filter
        if type_filter == "img" and not is_img: continue
        if type_filter == "ph" and not is_ph: continue

        # BPP Filter (Robust check via filename or sequence)
        if bpp_filter > 0:
            seq = run.get_product(pname)
            if seq.frame_config.bytes_per_pixel != bpp_filter:
                continue

        targets.append(pname)

    console.print(f"[bold green]Starting Filter Job[/]")
    console.print(f"Kernel: [cyan]{kernel_name}[/] | Params: {params}")
    console.print(f"Filters: Type={type_filter}, BPP={bpp_filter if bpp_filter else 'Any'}")
    console.print(f"Found {len(targets)} products.\n")

    if not targets:
        console.print("[yellow]No matching products found.[/]")
        return

    # 3. Process Loop
    stats_table = Table(title="Filter Statistics")
    stats_table.add_column("Product", style="cyan")
    stats_table.add_column("Original", justify="right")
    stats_table.add_column("Kept", justify="right", style="green")
    stats_table.add_column("Ratio", justify="right")
    stats_table.add_column("Time", justify="right")

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True
    ) as progress:

        main_task = progress.add_task("[bold]Processing Products...", total=len(targets))

        for prod_name in targets:
            seq = run.get_product(prod_name)
            try:
                res = _process_sequence(
                    seq, output_dir, batched_kernel_wrapper, params, batch_size, progress
                )
                stats_table.add_row(
                    seq.name, str(res['total']), str(res['kept']), f"{res['ratio']:.1f}%", f"{res['time']:.1f}s"
                )
            except Exception as e:
                logger.error(f"Failed {prod_name}: {e}")
                stats_table.add_row(seq.name, "ERROR", "-", "-", "-")

            progress.advance(main_task)

    console.print(stats_table)
    console.print(f"[bold green]Job Complete.[/] Output in: {output_dir}")


def _process_sequence(seq, out_dir, kernel_fn, params, batch_size, progress_ctx):
    # Construct Filename
    param_tag = ".".join([f"{k}_{v}" for k, v in params.items()])
    src_stem = seq.file_paths[0].name.split('.seqno')[0]
    out_name = f"{src_stem}.{param_tag}.seqno_0.pff"
    out_path = out_dir / out_name

    start_t = time.time()
    total_kept = 0

    task_id = progress_ctx.add_task(f"Filtering {seq.name}...", total=len(seq))

    with open(out_path, 'wb') as f_out:
        for chunk_start in range(0, len(seq), batch_size):
            img_batch = seq.get_image_array(chunk_start, batch_size)
            if len(img_batch) == 0: break

            try:
                # JAX Execution
                batch_results = kernel_fn(jnp.array(img_batch), **params)

                # Handle Return Types (Tuple vs Single Array)
                if isinstance(batch_results, (tuple, list)):
                    jax_keep = batch_results[-1]
                else:
                    jax_keep = batch_results

                keep_mask = np.array(jax_keep)
            except Exception as e:
                logger.error(f"JAX Error on chunk {chunk_start}: {e}")
                raise e

            # Write Kept Frames
            keep_indices = np.where(keep_mask)[0]
            if len(keep_indices) > 0:
                for local_idx in keep_indices:
                    global_idx = chunk_start + local_idx
                    raw_bytes = _fetch_raw_bytes(seq, global_idx)
                    f_out.write(raw_bytes)
                total_kept += len(keep_indices)

            progress_ctx.update(task_id, advance=len(img_batch))

    progress_ctx.remove_task(task_id)

    duration = time.time() - start_t
    ratio = (total_kept / len(seq)) * 100 if len(seq) > 0 else 0

    return {'total': len(seq), 'kept': total_kept, 'ratio': ratio, 'time': duration}


def _fetch_raw_bytes(seq, global_idx):
    file_idx = 0
    local_idx = global_idx
    for i, count in enumerate(seq._file_frame_counts):
        if local_idx < count:
            file_idx = i
            break
        local_idx -= count

    path = seq.file_paths[file_idx]
    offset = local_idx * seq.frame_config.frame_size

    with open(path, 'rb') as f:
        f.seek(offset)
        return f.read(seq.frame_config.frame_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PANOSETI Generic Filter")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--kernel", type=str, default="neighbor", help="Kernel name (neighbor, threshold)")
    parser.add_argument("--params", type=str, default="thresh=150,n_min=3", help="k=v,k2=v2")
    parser.add_argument("--product", type=str, default="all", help="Substring match for product name")
    parser.add_argument("--type", type=str, default="any", choices=["any", "img", "ph"], help="Filter by product type")
    parser.add_argument("--bpp", type=int, default=0, help="Filter by bytes-per-pixel (e.g. 2 for 16-bit)")

    args = parser.parse_args()

    run_filter_job(
        args.run_dir,
        args.output_dir,
        args.kernel,
        parse_params(args.params),
        args.product,
        args.type,
        args.bpp
    )