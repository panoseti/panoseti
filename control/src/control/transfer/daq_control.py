"""Client-side helpers for the panoseti_grpc daq_control service.

Wraps RPCs on ``panoseti_grpc.daq_control`` for use by both the transfer
daemon and `pseti xfr` CLI commands, so they share one implementation
instead of duplicating request-building and per-node fan-out/error-handling.
Currently only ``CleanupData`` is wrapped; add further daq_control RPC
wrappers here as they're needed.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from control.utils.util import daq_grpc_endpoint

if TYPE_CHECKING:
    from control.transfer.models import TransferJob, TransferNodeSpec

try:
    import panoseti_grpc.daq_control.client as daq_client
except ImportError:
    daq_client = None  # type: ignore


async def cleanup_daq_nodes(
    job: TransferJob,
    *,
    mode: str,
    delete_patterns: list[str] | None = None,
    preserve_patterns: list[str] | None = None,
) -> list[str]:
    """Call ``CleanupData`` on every DAQ node named in *job*, in parallel.

    Args:
        job: The transfer job whose ``daq_nodes`` to clean up.
        mode: ``"CLEANUP_FULL"`` (rmtree the whole run directory) or
            ``"CLEANUP_SELECTIVE"`` (delete only *delete_patterns*, keeping
            *preserve_patterns*). The server enforces a ``manifest_digest``
            precondition for ``CLEANUP_SELECTIVE``, so only use it once a
            manifest has actually been verified.
        delete_patterns: Glob patterns to delete (``CLEANUP_SELECTIVE`` only).
        preserve_patterns: Glob patterns to keep (``CLEANUP_SELECTIVE`` only).

    Returns:
        Human-readable error strings, one per node that failed to clean up;
        empty if every node succeeded. A single explanatory entry is
        returned if ``panoseti_grpc`` isn't installed.
    """
    if daq_client is None:
        return ["panoseti_grpc not available; skipping DAQ-node cleanup"]  # type: ignore[unreachable]

    errors: list[str] = []

    async def cleanup_node(node: TransferNodeSpec) -> None:
        host, port = daq_grpc_endpoint(node)
        request: dict[str, Any] = {
            "data_dir": node.data_dir,
            "run_dir": job.run_name,
            "module_id": node.module_ids,
            "mode": mode,
        }
        if delete_patterns is not None:
            request["delete_patterns"] = delete_patterns
        if preserve_patterns is not None:
            request["preserve_patterns"] = preserve_patterns
        try:
            async with daq_client.AsyncDaqControlClient(host=host, port=port) as client:
                resp = await asyncio.wait_for(client.CleanupData(request), timeout=15.0)
                if not resp.get("success", True):
                    errors.append(
                        f"CleanupData failed for {node.ip_addr}: {resp.get('message', 'unknown error')}"
                    )
        except Exception as exc:
            errors.append(f"CleanupData failed for {node.ip_addr}: {exc}")

    try:
        async with asyncio.TaskGroup() as tg:
            for node in job.daq_nodes:
                tg.create_task(cleanup_node(node))
    except* Exception as eg:
        for exc in eg.exceptions:
            errors.append(f"cleanup_node task failed: {exc}")

    return errors
