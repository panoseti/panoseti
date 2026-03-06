#! /usr/bin/env python3
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from panoseti_interface import PanosetiRun, PFFSequence
from jax_filters import KERNELS

# Constants
FRAME_SIZE = 2048  # Bytes per frame (32x32 * 2 bytes)
BATCH_SIZE = 2000


def bench_raw_io(filepath, n_frames=50000):
    """Test 1: Pure Disk Speed (Raw sequential reads, no parsing)."""
    print(f"\n--- Test 1: Raw Disk I/O ({filepath.name}) ---")

    file_size = filepath.stat().st_size
    # Don't try to read more than exists
    max_frames = file_size // FRAME_SIZE
    if max_frames == 0:
        print("File empty or too small.")
        return 0

    n_frames = min(n_frames, max_frames)
    total_bytes = n_frames * FRAME_SIZE

    start = time.perf_counter()
    with open(filepath, 'rb') as f:
        chunk_size = BATCH_SIZE * FRAME_SIZE
        bytes_read = 0
        while bytes_read < total_bytes:
            data = f.read(chunk_size)
            if not data: break
            bytes_read += len(data)

    dt = time.perf_counter() - start
    # Avoid div/0
    if dt == 0: dt = 0.000001

    fps = n_frames / dt
    mbps = (bytes_read / 1024 / 1024) / dt

    print(f"Time: {dt:.4f}s | FPS: {fps:,.0f} | Speed: {mbps:.1f} MB/s")
    return fps


def bench_parsing(seq, n_frames=50000):
    """Test 2: Parsing Overhead (PFFSequence -> Numpy Array)."""
    print(f"\n--- Test 2: PFF Parsing (seeking & restructuring) ---")

    n_frames = min(n_frames, len(seq))
    if n_frames == 0:
        print("Sequence empty.")
        return 0

    start = time.perf_counter()

    # Read in batches
    for i in range(0, n_frames, BATCH_SIZE):
        # We cap the batch to not go over n_frames
        count = min(BATCH_SIZE, n_frames - i)
        if count <= 0: break
        _ = seq.get_image_array(i, count)

    dt = time.perf_counter() - start
    if dt == 0: dt = 0.000001

    fps = n_frames / dt

    print(f"Time: {dt:.4f}s | FPS: {fps:,.0f}")
    return fps


def bench_compute(kernel_name, n_frames=50000):
    """Test 3: Pure Compute (Synthetic data in RAM -> JAX)."""
    print(f"\n--- Test 3: Pure Compute (JAX {kernel_name} kernel) ---")

    # Generate synthetic data in RAM
    print("Generating synthetic data in RAM...")
    data = np.random.randint(0, 400, (BATCH_SIZE, 32, 32), dtype=np.int16)
    data[:, 15, 15] = 500
    jax_data = jnp.array(data)

    kernel_func = KERNELS[kernel_name]

    # Define execution wrapper to handle different return types
    def run_batch():
        res = jax.vmap(kernel_func, in_axes=(0, None, None))(jax_data, 300, 2)
        # Handle Tuple return (Neighbor) vs Array return (Threshold)
        if isinstance(res, tuple):
            return res[-1]  # Return last element for blocking
        return res

    # JIT Compile (warmup)
    print("Compiling JAX kernel...")
    _res = run_batch()
    _res.block_until_ready()

    iters = max(1, n_frames // BATCH_SIZE)
    actual_frames = iters * BATCH_SIZE

    start = time.perf_counter()

    # Main Compute Loop
    for _ in range(iters):
        res_obj = run_batch()
        res_obj.block_until_ready()

    dt = time.perf_counter() - start
    if dt == 0: dt = 0.000001

    fps = actual_frames / dt

    print(f"Time: {dt:.4f}s | FPS: {fps:,.0f}")
    return fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--frames", type=int, default=100000)
    args = parser.parse_args()

    # Setup
    try:
        run = PanosetiRun(args.run_dir)
        # Find first image product
        prods = [p for p in run.list_products() if 'img' in p or 'ph' in p]
        if not prods:
            print("No image products found.")
            exit()

        seq = run.get_product(prods[0])
        raw_file = seq.file_paths[0]

        # Run Benchmarks
        io_fps = bench_raw_io(raw_file, args.frames)
        parse_fps = bench_parsing(seq, args.frames)
        comp_fps = bench_compute("neighbor", args.frames)

        print("\n" + "=" * 40)
        print("SUMMARY")
        print(f"1. Disk Limit:      {io_fps:,.0f} FPS")
        print(f"2. Parsing Limit:   {parse_fps:,.0f} FPS")
        print(f"3. Compute Limit:   {comp_fps:,.0f} FPS")
        print("=" * 40)

        bottleneck = min(v for v in [io_fps, parse_fps, comp_fps] if v > 0)
        print(f"Predicted Pipeline Max: {bottleneck:,.0f} FPS")

    except KeyboardInterrupt:
        print("\nAborted.")
    except Exception as e:
        print(f"\nError: {e}")