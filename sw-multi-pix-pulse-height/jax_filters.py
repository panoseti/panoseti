import jax
import jax.numpy as jnp
import logging

# Silence startup warnings
jax.config.update("jax_platform_name", "cpu")
logging.getLogger("jax").setLevel(logging.ERROR)

@jax.jit
def neighbor_kernel(raw_img, thresh, n_min):
    """
    O(1) Complexity Filter (Independent of trigger count).
    Logic:
      1. Identify Triggers (Raw > Thresh)
      2. Count Neighbors for every pixel using convolution/shifting.
      3. Keep if enough pixels are 'supported' by neighbors.
    """
    # 1. Thresholding
    trigger_mask = raw_img >= thresh
    total_triggers = jnp.sum(trigger_mask)

    # 2. Neighbor Check (3x3 Box)
    # We shift the image in 8 directions to check neighbors
    # This is equivalent to a convolution but often faster/simpler in pure JAX
    padded = jnp.pad(trigger_mask, 1, mode='constant')

    # Cardinals
    n_up = padded[:-2, 1:-1]
    n_down = padded[2:, 1:-1]
    n_left = padded[1:-1, :-2]
    n_right = padded[1:-1, 2:]

    # Diagonals (Add these)
    n_ul = padded[:-2, :-2]  # Up-Left
    n_ur = padded[:-2, 2:]  # Up-Right
    n_dl = padded[2:, :-2]  # Down-Left
    n_dr = padded[2:, 2:]  # Down-Right

    # Union of all 8
    has_neighbor = (n_up | n_down | n_left | n_right | n_ul | n_ur | n_dl | n_dr)
    supported_mask = trigger_mask & has_neighbor

    n_supported = jnp.sum(supported_mask)

    # 3. Decision
    # We keep if we have enough "clumped" pixels
    keep = n_supported >= n_min

    return trigger_mask, supported_mask, total_triggers, n_supported, keep

@jax.jit
def threshold_kernel(raw_img, thresh, n_min, **kwargs):
    return jnp.sum(raw_img >= thresh) >= n_min

# --- Kernel Registry ---
KERNELS = {
    "neighbor": neighbor_kernel,
    "threshold": threshold_kernel
}
