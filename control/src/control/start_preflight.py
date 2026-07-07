"""
Pre-flight validation checks for `pseti start`.

Split out of start.py: these are pure validation/reachability checks with
no transaction or CLI coupling, so they're independently unit-testable.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING

from panoseti_grpc.daq_data.client import AioDaqDataClient
from panoseti_grpc.telemetry.logger import get_logger

from control.interfaces import NetworkClient
from control.utils import config_file, util
from control.utils.paths import PanoPaths
from control.utils.run_state import ValidationError

if TYPE_CHECKING:
    from control.utils.pydantic_config_models import DaqConfig, DaqNode, NetworkConfig, QuaboUids

log_dir = PanoPaths.logs_dir()
log_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger(
    "PSETI.Start",
    log_dir=log_dir,
    grpc_enabled=True,
    reset=True
)


# check that PH calibration file is present, nonempty, and at most 24 hours old
#
def ph_baseline_file_ok(path: pathlib.Path | str | None = None) -> bool:
    """Verify that the Pulse Height calibration file is valid.

    Checks that the file exists, is not empty, and is at most 24 hours old.
    Stale or missing calibration data can lead to incorrect PH measurements.

    Args:
        path: Optional Path (or str) to the baseline file. Defaults to using config_file.get_quabo_ph_baselines().

    Returns:
        True if the file is valid and values are within range, False otherwise.
    """
    # Resolve the path if not provided
    if path is None:
        try:
            path = config_file.get_quabo_ph_baseline_path()
        except FileNotFoundError:
            print('PH baseline file not found.  Run config.py --calibrate-ph')
            return False

    path = pathlib.Path(path)
    if not path.exists():
        print(f'{path} not found.  Run config.py --calibrate-ph')
        return False

    # Common checks for the resolved path
    if path.stat().st_size == 0:
        print(f'{path} is empty.  Run config.py --calibrate-ph')
        return False
    if path.stat().st_mtime < time.time() - 86400:
        print(f'{path} is too old (>24h).  Run config.py --calibrate-ph')
        return False

    try:
        # Load and validate content
        with open(path) as f:
            data = json.load(f)
        from control.utils.pydantic_config_models import PhBaselineConfig
        baselines = PhBaselineConfig(**data)
        return config_file.validate_ph_baselines(baselines)
    except Exception as e:
        print(f"PH baseline validation failed: {e}")
        return False


@dataclass(frozen=True)
class QuaboProbeResult:
    uid: str
    ip: str
    port: int
    reachable: bool
    error: str | None

async def _quabo_reachability_report(
    quabo_uids: QuaboUids,
    network_config: NetworkConfig
) -> list[QuaboProbeResult]:
    """Perform a structured reachability sweep and return results per-Quabo."""
    results: list[QuaboProbeResult] = []

    async def check_one(uid: str, base_ip: str, index: int) -> None:
        from ipaddress import ip_address
        ip_ports = util.get_quabo_ip_port(ip_address(base_ip), index, network_config)
        real_ip = ip_ports.ip_addr
        cmd_port = ip_ports.cmd_port

        from control.utils.config_validator import _check_reachability

        loop = asyncio.get_running_loop()
        ok, err = await loop.run_in_executor(None, lambda: _check_reachability(str(real_ip), cmd_port, target_type="quabo", timeout=2.0))
        results.append(QuaboProbeResult(
            uid=uid,
            ip=str(real_ip),
            port=cmd_port,
            reachable=ok,
            error=err if not ok else None
        ))

    async with asyncio.TaskGroup() as tg:
        for dome in quabo_uids.domes:
            for module in dome.modules:
                base_ip = str(module.ip_addr)
                for i in range(4):
                    uid = module.quabos[i].uid
                    if uid != '':
                        tg.create_task(check_one(uid, base_ip, i))
    return results

async def _check_quabo_reachability(
    quabo_uids: QuaboUids,
    network_config: NetworkConfig,
    lenient: bool = False
) -> None:
    """Verify that all configured Quabos are reachable on the network."""
    logger.info("Performing Quabo reachability sweep...")

    results = await _quabo_reachability_report(quabo_uids, network_config)
    unreachable = [r for r in results if not r.reachable]

    if not unreachable:
        logger.info("All configured Quabos are reachable.")
        return

    for r in unreachable:
        msg = f"Quabo {r.uid} at {r.ip}:{r.port} is UNREACHABLE: {r.error}"
        if lenient:
            logger.warning(f"{msg} (Non-fatal in container/CI environment)")
        else:
            logger.error(msg)

    if not lenient:
        summary = "\n".join([f"  - {r.uid} ({r.ip}:{r.port}): {r.error}" for r in unreachable])
        raise ValidationError(f"One or more Quabos are unreachable:\n{summary}")


async def _check_daq_reachability(daq_config: DaqConfig, net_client: NetworkClient) -> None:
    """Verify that all configured DAQ nodes are responding via gRPC."""
    logger.info("Performing DAQ node gRPC reachability sweep...")
    async def check_node_grpc(node: DaqNode) -> None:
        if not node.module_ids:
            return
        try:
            await net_client.get_daq_status(node, timeout_s=5.0)
        except Exception as e:
            raise ValidationError(f"DAQ node {node.ip_addr} gRPC is unreachable: {e}") from e

    try:
        async with asyncio.TaskGroup() as tg:
            for node in daq_config.daq_nodes:
                tg.create_task(check_node_grpc(node))
        logger.info("All configured DAQ nodes are reachable via gRPC.")
    except* Exception as eg:
        for i, exc in enumerate(eg.exceptions, 1):
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            logger.error(f"gRPC reachability check {i}/{len(eg.exceptions)} failed: {type(exc).__name__}: {exc}\n{tb}")
        raise ValidationError("One or more DAQ nodes are unreachable via gRPC.") from eg


def _resolve_strict_mode(strict_flag: bool | None, daq_config: DaqConfig) -> bool:
    """Resolve the effective strict mode for pre-flight checks.

    Resolution order:
    1. CLI flag ``--strict/--no-strict`` (highest priority).
    2. Env var ``PSETI_STRICT=1|0``.
    3. Tier-aware default: ``True`` unless we are in a pure-SW container CI
       tier (tier3_fleet, tier4_chaos, tier5_integration).  HW-SW tests and
       bare-metal runs always default to strict.

    Args:
        strict_flag: Value from ``--strict/--no-strict`` CLI option.  ``None``
            means the flag was not passed and the default should be applied.
        daq_config: Loaded DAQ configuration model.

    Returns:
        True if strict mode is active; False for lenient mode.
    """
    if strict_flag is not None:
        return strict_flag
    env_val = os.environ.get("PSETI_STRICT")
    if env_val is not None:
        return env_val.strip() not in ("0", "false", "no")
    # Default: lenient only in known SW-simulation CI tiers.
    sw_tiers = {"tier3_fleet", "tier4_chaos", "tier5_integration"}
    in_sw_tier = os.environ.get("PSETI_TEST_TIER", "") in sw_tiers
    return not (daq_config.head_node_container and in_sw_tier)


async def _check_no_remote_hashpipe(daq_config: DaqConfig, net_client: NetworkClient, force_restart: bool = False) -> None:
    """Verify that no DAQ node is already running Hashpipe.

    Prevents a new start transaction from issuing UDP configuration to Quabos
    while an existing observation is in progress, which would corrupt the
    running run.

    Args:
        daq_config: Validated DAQ configuration model.
        net_client: Dependency-injected network client.
        force_restart: If True, call StopDaq on each node that reports a
            running Hashpipe instead of raising.

    Raises:
        ValidationError: If any DAQ node reports Hashpipe running and
            *force_restart* is False.
    """
    logger.info("Performing remote Hashpipe liveness pre-flight check...")

    async def check_node(node: DaqNode) -> None:
        if not node.module_ids:
            return
        try:
            status = await net_client.get_daq_status(node, timeout_s=5.0)
            if hasattr(status, 'hashpipe_running'):
                running = status.hashpipe_running
                pid = status.hashpipe_pid
            else:
                running = status.get("hashpipe_running")
                pid = status.get("hashpipe_pid")
        except Exception as e:
            logger.warning("StatusDaq failed for %s during hashpipe pre-flight: %s", node.ip_addr, e)
            return

        if running:
            if force_restart:
                logger.warning(
                    "Hashpipe running on %s (pid=%s) — stopping per --force-restart.",
                    node.ip_addr, pid,
                )
                try:
                    await net_client.stop_daq_node(node, timeout_s=15.0)
                except Exception as stop_err:
                    raise ValidationError(
                        f"--force-restart: StopDaq failed for {node.ip_addr}: {stop_err}"
                    ) from stop_err
            else:
                raise ValidationError(
                    f"Hashpipe already running on {node.ip_addr} "
                    f"(pid={pid}). "
                    "Run `pseti stop` first, or pass --force-restart."
                )

    try:
        async with asyncio.TaskGroup() as tg:
            for node in daq_config.daq_nodes:
                tg.create_task(check_node(node))
    except* Exception as eg:
        for i, exc in enumerate(eg.exceptions, 1):
            if isinstance(exc, ValidationError):
                 logger.error(f"Remote Hashpipe check {i}/{len(eg.exceptions)} failed: {exc}")
            else:
                 tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                 logger.error(f"Remote Hashpipe check {i}/{len(eg.exceptions)} failed: {type(exc).__name__}: {exc}\n{tb}")

        # If any were ValidationErrors, re-raise the first one to maintain existing API contract
        # but the operator now sees all of them in the log.
        for exc in eg.exceptions:
            if isinstance(exc, ValidationError):
                raise exc from None
        raise ValidationError("Remote Hashpipe liveness check failed.") from eg

    logger.info("Remote Hashpipe pre-flight OK — no running instances detected.")


async def _check_daq_data_status(
    daq_config: DaqConfig,
    network_config: NetworkConfig,
    do_init: bool = False,
    gateway_host: str = "localhost",
    gateway_port: int | None = None,
) -> None:
    """Verify DaqData gateway is reachable and optionally request re-initialization."""
    if gateway_port is None:
        gateway_port = int(os.getenv("DAQ_DATA_GATEWAY_PORT", "50051"))
        
    logger.info("Performing DaqData gateway status pre-flight check...")

    async with AioDaqDataClient(gateway_host, gateway_port) as client:
        status = await client.status()
        if not status or not status.hp_io_initialized:
            if do_init:
                logger.info("Requesting DaqData gateway re-initialization...")
                # The gateway proxies InitHpIo to each edge node as-is, so
                # data_dir must be a real, non-empty path (InitHpIoParameters
                # requires it) even though edge nodes normally auto-init on
                # their own startup and ignore this unless force=True.
                hp_io_cfg = {
                    "data_dir": daq_config.daq_nodes[0].data_dir,
                    "update_interval_seconds": 0.1,
                    "force": False,
                    "simulate_daq": False,
                    "module_ids": [],
                }
                success = await client.init_hp_io(hp_io_cfg)
                if success:
                    logger.info("DaqData gateway re-initialization succeeded.")
                else:
                    logger.warning("DaqData gateway re-initialization failed.")
            else:
                logger.warning(
                    "DaqData gateway hp_io is NOT initialized. "
                    "Real-time streaming (pseti show sci) will not be available. "
                    "Use --init-snapshot to enable it automatically (default)."
                )
        else:
            logger.info("DaqData gateway hp_io is initialized and valid.")
