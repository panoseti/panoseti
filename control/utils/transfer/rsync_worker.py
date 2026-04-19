from __future__ import annotations

import subprocess
import time

_TRANSIENT_CODES = {12, 23, 30, 35, 255}
_MAX_RETRIES = 3
_BASE_DELAY = 5


def _run_rsync_with_retry(cmd: list[str], verbose: bool = False) -> str:
    """Run rsync with exponential backoff for transient failures.

    Args:
        cmd: Full rsync command as a list of strings.
        verbose: If True, prints the command before each attempt.

    Returns:
        Empty string on success, or an error message describing the failure.
    """
    for attempt in range(_MAX_RETRIES + 1):
        if verbose:
            print(" ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            return ""
        if res.returncode in _TRANSIENT_CODES and attempt < _MAX_RETRIES:
            delay = _BASE_DELAY * (2**attempt)
            print(
                f"Transient rsync error {res.returncode}. "
                f"Retrying in {delay}s (attempt {attempt + 1}/{_MAX_RETRIES})..."
            )
            time.sleep(delay)
            continue
        err_type = "Transient" if res.returncode in _TRANSIENT_CODES else "Terminal"
        return f"{err_type} rsync error {res.returncode}: {res.stderr}"
    return "Max retries exceeded"


def rsync_one_node(
    node_ip: str,
    node_data_dir: str,
    run_name: str,
    head_data_dir: str,
    username: str = "panoseti",
    port_forwarding: dict | None = None,
    verbose: bool = False,
) -> tuple[bool, str]:
    """Rsync a single DAQ node's run directory to the head node.

    Pulls the per-run stdout/snapshot files and all per-module PFF data into
    the matching head-node run directory.  The rsync call mirrors the combined
    invocation used by ``utils.file_xfer.copy_dir_from_node``.

    Args:
        node_ip: IP address of the remote DAQ node.
        node_data_dir: Absolute data directory path on the remote node.
        run_name: Name of the run directory to transfer.
        head_data_dir: Absolute data directory on the head node (destination).
        username: SSH username for the remote node (default ``panoseti``).
        port_forwarding: Optional dict with keys ``status`` (bool), ``port``
            (int), and ``gw_ip`` (str) for gateway port-forwarding.  When
            ``status`` is True the rsync is tunnelled through the gateway.
        verbose: If True, prints the rsync command before execution.

    Returns:
        A ``(success, error_message)`` tuple.  ``error_message`` is empty on
        success.
    """
    import os

    run_dir_path = os.path.join(head_data_dir, run_name)
    if not os.path.isdir(run_dir_path):
        return False, f"rsync_one_node(): no run dir {run_dir_path}"

    use_pf = (
        port_forwarding is not None
        and port_forwarding.get("status", False)
    )

    base_rsync: list[str] = ["rsync", "-P"]
    if use_pf and port_forwarding is not None:
        base_rsync.extend(["-e", f"ssh -p {port_forwarding['port']}"])
        remote_host = f"{username}@{port_forwarding['gw_ip']}"
    else:
        remote_host = f"{username}@{node_ip}"

    # Import prefix constants from util to stay consistent with file_xfer.py.
    try:
        from utils.util import hp_stdout_prefix, pss_prefix
    except ImportError:
        hp_stdout_prefix = "hp_stdout"
        pss_prefix = "pss_"

    cmd = [
        *base_rsync,
        f"{remote_host}:{node_data_dir}/{run_name}/{hp_stdout_prefix}*",
        f"{remote_host}:{node_data_dir}/{run_name}/{pss_prefix}*",
        # Wildcard picks up all module_N subdirs for this run in one shot.
        f"{remote_host}:{node_data_dir}/module_*/{run_name}/*",
        run_dir_path,
    ]
    err = _run_rsync_with_retry(cmd, verbose)
    if err:
        return False, f"rsync_one_node({node_ip}): {err}"
    return True, ""
