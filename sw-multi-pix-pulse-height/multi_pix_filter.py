import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import logging
from rich.logging import RichHandler

# Keep existing imports from previous panoseti_interface...
from panoseti_interface import PanosetiRun, PFFSequence

# Setup Logger
logging.getLogger("jax").setLevel(logging.ERROR)  # Suppress JAX startup noise
logger = logging.getLogger("JAXProcessor")
logger.setLevel(logging.INFO)
logger.addHandler(RichHandler())


class JAXFrameProcessor:
    """
    Accelerated processor for PANOSETI frames using JAX.
    Handles calibration and geometric filtering on the GPU/TPU/CPU.
    """

    def __init__(self, shape=(16, 16)):
        self.shape = shape
        # Pre-compute coordinate grids for CoM calculations
        # This is static and pushed to the device once.
        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        self.grid_x = jnp.array(x, dtype=jnp.float32)
        self.grid_y = jnp.array(y, dtype=jnp.float32)

        # JIT-compile the processing kernel immediately
        self.process_batch = jax.jit(self._core_kernel)

        # Calibration matrices (start as Identity, updated via setters)
        self.B = jnp.zeros(shape, dtype=jnp.float32)
        self.G = jnp.ones(shape, dtype=jnp.float32)

    def set_calibration(self, baseline: np.ndarray, gain: np.ndarray):
        """Uploads calibration matrices to the JAX device."""
        self.B = jnp.array(baseline, dtype=jnp.float32)
        self.G = jnp.array(gain, dtype=jnp.float32)
        # Avoid division by zero in gain
        self.G = jnp.where(self.G == 0, 1.0, self.G)

    @partial(jax.jit, static_argnums=(0,))
    def _core_kernel(self,
                     raw_batch,
                     baseline,
                     gain,
                     grid_x,
                     grid_y,
                     threshold,
                     min_pixels,
                     radius,
                     min_k_in_radius):
        """
        The fused JAX Kernel.
        Args:
            raw_batch: (B, H, W) raw ADC data
        Returns:
            pe_batch: (B, H, W) calibrated data
            keep_mask: (B,) boolean array (True = keep frame)
        """
        # 1. Calibration (ADC -> PE)
        # PE = (ADC - B) / G
        # Broadcasting B and G over the batch dimension
        pe_batch = (raw_batch - baseline) / gain

        # 2. Filtering Logic
        # Mask of pixels above threshold
        # shape: (B, H, W)
        trigger_mask = pe_batch > threshold

        # Count pixels above threshold per frame
        # shape: (B,)
        trigger_counts = jnp.sum(trigger_mask, axis=(1, 2))

        # --- Center of Mass Calculation ---
        # We need to handle the case where sum is 0 (division by zero)
        safe_sum = jnp.where(trigger_counts == 0, 1.0, trigger_counts)

        # Weighted sum of coordinates (weights are 1 where triggered, 0 otherwise)
        # shape: (B,)
        sum_x = jnp.sum(grid_x * trigger_mask, axis=(1, 2))
        sum_y = jnp.sum(grid_y * trigger_mask, axis=(1, 2))

        com_x = sum_x / safe_sum
        com_y = sum_y / safe_sum

        # --- Radius Check ---
        # Reshape CoM for broadcasting: (B, 1, 1)
        com_x = com_x[:, None, None]
        com_y = com_y[:, None, None]

        # Calculate squared distance of ALL pixels to the calculated CoM
        # dist_sq: (B, H, W)
        dist_sq = (grid_x - com_x) ** 2 + (grid_y - com_y) ** 2
        radius_sq = radius ** 2

        # Check: Is pixel triggered AND within radius?
        # logic: (Distance < r) AND (Pixel > Threshold)
        compact_mask = (dist_sq <= radius_sq) & trigger_mask

        # Count pixels satisfying compactness
        compact_counts = jnp.sum(compact_mask, axis=(1, 2))

        # --- Final Decision ---
        # Condition A: n pixels > threshold
        pass_a = trigger_counts >= min_pixels

        # Condition B: k pixels within radius d
        pass_b = compact_counts >= min_k_in_radius

        final_decision = pass_a & pass_b

        return pe_batch, final_decision

    def run(self, raw_numpy_batch: np.ndarray, params: dict):
        """
        Public execution method.
        params must contain: 'threshold', 'n_pixels', 'radius', 'k_compact'
        """
        # Convert inputs to JAX arrays (moves data to GPU if available)
        raw_jax = jnp.array(raw_numpy_batch, dtype=jnp.float32)

        pe_batch, decisions = self.process_batch(
            raw_jax,
            self.B,
            self.G,
            self.grid_x,
            self.grid_y,
            params['threshold'],
            params['n_pixels'],
            params['radius'],
            params['k_compact']
        )

        # Block until computation is done and return as numpy
        return np.array(pe_batch), np.array(decisions)


# --- Usage Example ---

if __name__ == "__main__":
    # 1. Setup Data Interface
    run_dir = "palomar_data/obs_Palomar.start_2026-01-20T02:24:17Z.runtype_obs-test.pffd"
    run = PanosetiRun(run_dir)

    # Get a product (e.g. ph1024 - which is actually 32x32, usually handled as 4 quabos)
    # Let's assume we are processing a Quabo stream (16x16)
    # If using Module data (32x32), change shape in processor init
    seq = run.get_product("dp_ph256.bpp_2.module_253")

    # 2. Setup JAX Processor
    # Using 16x16 for ph256
    processor = JAXFrameProcessor(shape=(16, 16))

    # Mock Calibration (In real usage, load using CalibrationManager from previous turn)
    dummy_baseline = np.full((16, 16), 280.0)
    dummy_gain = np.full((16, 16), 60.0)
    processor.set_calibration(dummy_baseline, dummy_gain)

    # 3. Filter Parameters
    filter_params = {
        'threshold': 5.0,  # PE units
        'n_pixels': 3,  # Condition A: At least 3 pixels > 5 PE
        'radius': 2.5,  # Distance in pixels
        'k_compact': 2  # Condition B: At least 2 triggering pixels within radius 2.5 of CoM
    }

    logger.info("Starting JAX Batch Processing...")

    # Process in Batches (e.g., 10,000 frames at a time)
    BATCH_SIZE = 10000
    total_found = 0

    for i in range(0, len(seq), BATCH_SIZE):
        # A. Read Raw Data (IO Bound)
        # get_image_array handles the mmap efficiency
        raw_chunk = seq.get_image_array(i, BATCH_SIZE)

        if raw_chunk.shape[0] == 0: break

        # B. Process on Device (Compute Bound - Accelerated)
        calibrated, mask = processor.run(raw_chunk, filter_params)

        # C. Handle Results
        n_hits = np.sum(mask)
        if n_hits > 0:
            logger.info(f"Batch {i}: Found {n_hits} interesting events.")

            # Extract interesting frames
            hits = calibrated[mask]

            # (Optional) Do something with hits, like saving to a new PFF
            # ...

        total_found += n_hits

    logger.info(f"Processing complete. Total events: {total_found}")