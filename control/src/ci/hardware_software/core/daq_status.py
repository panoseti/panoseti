"""
DAQ node status helpers.

Uses the gRPC DaqControl service (GetTransferStatus) to measure filesystem
free space before and after a window, confirming Hashpipe is writing data.
Also uses StatusDaq to check Hashpipe's live thread count: a PID existing is
not sufficient evidence Hashpipe is actually working -- it can block forever
inside hashpipe_databuf_create() during shared-memory/semaphore init without
ever spawning net_thread/compute_thread/output_thread (see
panoseti_grpc.daq_control.util.cleanup_stale_hashpipe_semaphores).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from control.utils.pydantic_config_models import DaqConfig, DaqNode

logger = logging.getLogger(__name__)


def _free_bytes(host: str, port: int, data_dir: str, run_name: str) -> int:
    """Return filesystem free bytes on the DAQ node, or -1 on error."""
    try:
        from panoseti_grpc.daq_control.client import DaqControlClient
        client = DaqControlClient(host=host, port=port)
        try:
            result = client.GetTransferStatus(
                {"data_dir": data_dir, "run_dir": run_name},
                timeout=10.0,
            )
            return int(result.get("free_bytes", -1))
        finally:
            client.close()
    except Exception as exc:
        logger.warning("[DAQ-STATUS] GetTransferStatus failed (%s:%d): %s", host, port, exc)
        return -1


def assert_disk_growing(
    node: DaqNode,
    daq_config: DaqConfig,
    run_name: str,
    min_bytes: int = 500_000,
    window_s: float = 10.0,
) -> None:
    """Assert that Hashpipe is actively writing data on the DAQ node.

    Measures filesystem free space before and after *window_s* seconds.
    If free space decreased by at least *min_bytes*, data is being written.

    Args:
        node: The DaqNode model (supplies data_dir and IP address).
        daq_config: Full DaqConfig (used to resolve the correct gRPC endpoint).
        run_name: Current run name (for logging).
        min_bytes: Minimum bytes written expected over the window.
        window_s: Measurement window in seconds.

    Raises:
        AssertionError: If the disk is not growing fast enough.
    """
    from control.utils.util import daq_grpc_endpoint

    host, port = daq_grpc_endpoint(node, daq_config)
    data_dir = node.data_dir

    free_before = _free_bytes(host, port, data_dir, run_name)
    logger.info(
        "[DAQ-STATUS] %s:%d run=%s free_before=%d bytes",
        host, port, run_name, free_before,
    )

    time.sleep(window_s)

    free_after = _free_bytes(host, port, data_dir, run_name)
    logger.info(
        "[DAQ-STATUS] %s:%d run=%s free_after=%d bytes",
        host, port, run_name, free_after,
    )

    if free_before < 0 or free_after < 0:
        raise AssertionError(
            f"DAQ node {host}:{port} GetTransferStatus failed "
            f"(free_before={free_before}, free_after={free_after}). "
            "Is the gRPC server reachable?"
        )

    delta_used = free_before - free_after
    assert delta_used >= min_bytes, (
        f"DAQ node {host}:{port} not writing fast enough: "
        f"only {delta_used} bytes freed in {window_s:.0f}s "
        f"(expected ≥ {min_bytes}). Is Hashpipe running for run={run_name!r}?"
    )
    logger.info(
        "[DAQ-STATUS] disk growing OK: %d bytes written in %.0fs for run=%s",
        delta_used, window_s, run_name,
    )


def assert_hashpipe_healthy(
    node: DaqNode,
    daq_config: DaqConfig,
    run_name: str,
) -> None:
    """Assert Hashpipe is not just running, but past its stuck-at-init window.

    A live PID is not sufficient evidence: Hashpipe can block forever inside
    hashpipe_databuf_create() during shared-memory/semaphore init -- before
    it ever spawns net_thread/compute_thread/output_thread -- and look
    identical to a healthy process under a plain PID-existence check. This
    calls StatusDaq and checks the thread-count-derived hashpipe_healthy
    flag (server-side default is unhealthy only when running with a thread
    count below EXPECTED_HASHPIPE_THREADS).

    Args:
        node: The DaqNode model (supplies data_dir and IP address).
        daq_config: Full DaqConfig (used to resolve the correct gRPC endpoint).
        run_name: Current run name (for logging).

    Raises:
        AssertionError: If Hashpipe is not running, or is running but unhealthy.
    """
    from control.utils.util import daq_grpc_endpoint
    from panoseti_grpc.daq_control.client import DaqControlClient

    host, port = daq_grpc_endpoint(node, daq_config)

    client = DaqControlClient(host=host, port=port)
    try:
        result = client.StatusDaq(
            {"data_dir": node.data_dir, "check_hashpipe_running": True,
             "check_disk_usage": False, "check_run_dirs": False},
            timeout=10.0,
        )
    finally:
        client.close()

    running = result.get("hashpipe_running")
    pid = result.get("hashpipe_pid")
    thread_count = result.get("hashpipe_thread_count", 0)
    healthy = result.get("hashpipe_healthy", True)

    logger.info(
        "[DAQ-STATUS] %s:%d run=%s hashpipe_running=%s pid=%s thread_count=%s healthy=%s",
        host, port, run_name, running, pid, thread_count, healthy,
    )

    assert running, (
        f"DAQ node {host}:{port} reports Hashpipe not running for run={run_name!r}."
    )
    assert healthy, (
        f"DAQ node {host}:{port} Hashpipe (pid={pid}) is stuck at {thread_count} "
        f"thread(s) for run={run_name!r} -- net_thread/compute_thread/output_thread "
        "never fully came up. Likely a leaked shared-memory semaphore from a prior "
        "process that was killed rather than stopped cleanly; retry `pseti start "
        "--force-clean-semaphores`."
    )
