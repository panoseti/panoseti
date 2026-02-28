import jax
import jax.numpy as jnp
import logging

# Silence startup warnings
jax.config.update("jax_platform_name", "cpu")
logging.getLogger("jax").setLevel(logging.ERROR)

@jax.jit
def neighbor_kernel(raw_img, thresh, n_min):
    """
     neighbor support filter -> compute independent of trigger count

      1. identify triggers (raw > thresh)
      2. count neighbors for every pixel using convolution/shifting.
      3. keep if enough pixels are supported by neighbors.
    """
    #thresholding
    trigger_mask = raw_img >= thresh
    total_triggers = jnp.sum(trigger_mask)

    # b neighbor check (3x3 Box)
    #  shift the image in 8 directions to check neighbors
    #  equiv to a convolution but more eff in JAX
    padded = jnp.pad(trigger_mask, 1, mode='constant')

    # cardinals
    n_up = padded[:-2, 1:-1]
    n_down = padded[2:, 1:-1]
    n_left = padded[1:-1, :-2]
    n_right = padded[1:-1, 2:]

    # diagonals
    n_ul = padded[:-2, :-2]  # up left
    n_ur = padded[:-2, 2:]  # up right
    n_dl = padded[2:, :-2]  # down left
    n_dr = padded[2:, 2:]  # down right

    # union of all 8
    has_neighbor = (n_up | n_down | n_left | n_right | n_ul | n_ur | n_dl | n_dr)
    supported_mask = trigger_mask & has_neighbor

    n_supported = jnp.sum(supported_mask)

    # c. decision
    # keep if we have enough supported pixels
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
