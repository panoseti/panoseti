"""
Checks for validated data products on the headnode.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console

from control.utils.panoseti_interface import PanosetiRun

logger = logging.getLogger(__name__)
console = Console()

def verify_data_products(
    run_path: Path, 
    num_frames_to_read: int = 10, 
    num_headers_to_print: int = 4
) -> None:
    """
    Validate that recorded data products are readable and well-formed.
    
    1. Scans the run directory.
    2. Displays the run structure (configs + products).
    3. For each non-empty product:
       - Reads first N frames (validates mmap/JSON/payload).
       - Prints first M headers.
       - Asserts image structure correctness.
    """
    logger.info("Verifying data products at: %s", run_path)
    pseti_run = PanosetiRun(run_path)
    
    # Show run structure
    pseti_run.show()
    
    products = pseti_run.list_products()
    if not products:
        raise ValueError(f"No data products found in {run_path}")
    
    for prod_name in products:
        seq = pseti_run.get_product(prod_name)
        n_frames = len(seq)
        
        if n_frames == 0:
            logger.info("Product %s is empty, skipping.", prod_name)
            continue
            
        frames_to_check = min(num_frames_to_read, n_frames)
        logger.info("Reading first %d frames of %s (total: %d)", 
                    frames_to_check, prod_name, n_frames)
        
        for i in range(frames_to_check):
            header, image = seq.get_frame(i)
            
            # Print headers for manual inspection (at most num_headers_to_print)
            if i < num_headers_to_print:
                console.print(f"[bold cyan]Header for {prod_name} Frame {i}:[/]")
                console.print(header)
                
            # Structural Assertions
            assert image.shape == seq.frame_config.image_shape, \
                f"{prod_name} frame {i} shape mismatch: {image.shape} != {seq.frame_config.image_shape}"
            assert image.dtype == seq.frame_config.dtype, \
                f"{prod_name} frame {i} dtype mismatch: {image.dtype} != {seq.frame_config.dtype}"

    logger.info("Data product verification complete for %s", run_path.name)
