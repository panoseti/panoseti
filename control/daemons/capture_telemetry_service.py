#! /usr/bin/env python3
"""
capture_telemetry_service.py — Headnode Telemetry gRPC daemon.

Runs the Telemetry gRPC service via the unified panoseti-server framework.
DAQ nodes configured with grpc_logging=true forward their log records here;
the Telemetry service batches them into Redis (logs:ingress), from which
storeLoki.py ships them to Loki.

Config resolution order:
  1. capture_telemetry_service/server.toml  — site-specific server config
  2. bundled 'headnode' profile             — fallback, with env-var overrides

Environment variables:
  REDIS_HOST  — Redis hostname (applied when using bundled profile)
  GRPC_PORT   — override server port (always applied)
"""
import asyncio
import contextlib
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from panoseti_grpc.server import PanosetiServer, PanosetiServerConfig
    from panoseti_grpc.telemetry.resources import make_rich_logger
except ImportError:
    print("CRITICAL: 'panoseti_grpc' not installed.")
    sys.exit(1)

# Local config files alongside this daemon
LOCAL_SERVER_CONFIG    = Path(__file__).parent / "capture_telemetry_service" / "server.toml"
LOCAL_TELEMETRY_CONFIG = Path(__file__).parent / "capture_telemetry_service" / "telemetry_config.toml"

logger = make_rich_logger("telemetry_daemon")


def _build_config() -> PanosetiServerConfig:
    """Load config: local server.toml > bundled headnode profile > env overrides."""
    if LOCAL_SERVER_CONFIG.exists():
        logger.info(
            f"Using server config: [bold cyan]{LOCAL_SERVER_CONFIG}[/]",
            extra={"markup": True},
        )
        cfg = PanosetiServerConfig.from_toml(LOCAL_SERVER_CONFIG)
    else:
        logger.info("No local server.toml found; using bundled 'headnode' profile.")
        cfg = PanosetiServerConfig.load_profile("headnode")
        # Apply runtime overrides when falling back to the bundled profile
        cfg.telemetry.redis_host = os.getenv("REDIS_HOST", cfg.telemetry.redis_host)
        if LOCAL_TELEMETRY_CONFIG.exists():
            logger.info(f"Using telemetry device config: {LOCAL_TELEMETRY_CONFIG}")
            cfg.telemetry.telemetry_config_path = str(LOCAL_TELEMETRY_CONFIG)

    # Port override always applies (set by session_start.py via GRPC_PORT env var)
    cfg.port = int(os.getenv("GRPC_PORT", str(cfg.port)))
    return cfg


def main() -> None:
    cfg = _build_config()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(PanosetiServer.run(cfg))


if __name__ == "__main__":
    main()
