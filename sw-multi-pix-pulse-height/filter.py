#! /usr/bin/env python3
import argparse
import logging
import time
import psutil
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Any
from functools import partial

# Imports
try:
    from panoseti_interface import PanosetiRun, PFFSequence
    from jax_filters import KERNELS
    import orjson
except ImportError:
    print("Missing modules: Ensure panoseti_interface.py, jax_filters.py, orjson are installed.")
    exit(1)

# Rich
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

# Configuration
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
console = Console()
logging.basicConfig(level="INFO", handlers=[RichHandler(console=console, markup=True)])
logger = logging.getLogger("PanoFilter")


def parse_params(param_str: str) -> Dict[str, Any]:
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
        save_output: bool = False,
        batch_size: int = 2**13
):
    if not run_dir.exists():
        logger.error(f"Run dir not found: {run_dir}")
        return

    # Setup JAX
    kernel_func = KERNELS.get(kernel_name)
    if not kernel_func:
        logger.error(f"Kernel '{kernel_name}' not found.")
        return

    def batched_kernel_wrapper(batch, **p):
        bound_func = partial(kernel_func, **p)
        return jax.vmap(bound_func)(batch)

    # Load Run
    run = PanosetiRun(run_dir)
    run.show()  # Display tree

    if save_output:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Select Targets
    all_prods = run.list_products()
    targets = []
    for pname in all_prods:
        if product_filter != "all" and product_filter not in pname: continue
        is_img = "img" in pname
        is_ph = "ph" in pname
        if type_filter == "img" and not is_img: continue
        if type_filter == "ph" and not is_ph: continue
        targets.append(pname)

    # Stats Table
    stats_table = Table(title=f"Filter Job: {kernel_name}")
    stats_table.add_column("Product", style="cyan")
    stats_table.add_column("Frames", justify="right")
    stats_table.add_column("Kept", justify="right", style="green")
    stats_table.add_column("Speed (FPS)", justify="right")
    stats_table.add_column("Read (MB)", justify="right")

    with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), TimeRemainingColumn(), console=console
    ) as progress:

        main_task = progress.add_task("Processing...", total=len(targets))

        for prod_name in targets:
            seq = run.get_product(prod_name)
            p = psutil.Process()
            io_start = p.io_counters() if hasattr(p, 'io_counters') else None

            res = _process_sequence(seq, output_dir, batched_kernel_wrapper, params, batch_size, save_output)

            io_end = p.io_counters() if hasattr(p, 'io_counters') else None
            read_mb = (io_end.read_bytes - io_start.read_bytes) / 1e6 if (io_start and io_end) else 0.0

            stats_table.add_row(
                seq.name, f"{res['total']:,}", f"{res['kept']:,}",
                f"{res['fps']:,.0f}", f"{read_mb:.1f}"
            )
            progress.advance(main_task)

    console.print(stats_table)


def _process_sequence(seq, out_dir, kernel_fn, params, batch_size, save_output):
    f_out = None
    if save_output:
        param_tag = ".".join([f"{k}_{v}" for k, v in params.items()])
        out_name = f"{seq.name}.{param_tag}.filtered.pff"
        f_out = open(out_dir / out_name, 'wb')

    start_t = time.perf_counter()
    total_kept = 0
    total_frames = len(seq)

    try:
        for chunk_start in range(0, total_frames, batch_size):
            img_batch = seq.get_image_array(chunk_start, batch_size)
            if len(img_batch) == 0: break

            jax_batch = jnp.array(img_batch)
            batch_results = kernel_fn(jax_batch, **params)

            # Extract boolean mask (assuming last return element is mask)
            if isinstance(batch_results, (tuple, list)):
                keep_mask = np.array(batch_results[-1])
            else:
                keep_mask = np.array(batch_results)

            keep_indices = np.where(keep_mask)[0]
            count_kept = len(keep_indices)
            total_kept += count_kept

            if save_output and count_kept > 0:
                # Optimized write loop for kept frames
                for local_idx in keep_indices:
                    global_idx = chunk_start + local_idx
                    # Manually replicate raw read logic to avoid overhead
                    f_idx, f_local = seq._locate_frame(global_idx)
                    mm = seq._get_mmap(f_idx)
                    off = f_local * seq.frame_config.frame_size
                    end = off + seq.frame_config.frame_size
                    f_out.write(mm[off:end])
    finally:
        if f_out: f_out.close()

    dt = time.perf_counter() - start_t
    return {'total': total_frames, 'kept': total_kept, 'fps': total_frames / (dt if dt > 0 else 1e-6)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--kernel", default="neighbor")
    parser.add_argument("--params", default="thresh=125,n_min=15")
    parser.add_argument("--product", default="all")
    parser.add_argument("--type", default="any")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    run_filter_job(args.run_dir, args.output_dir, args.kernel, parse_params(args.params), args.product, args.type,
                   args.save)