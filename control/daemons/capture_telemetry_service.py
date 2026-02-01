#! /usr/bin/env python3
import os
import sys
import asyncio
from pathlib import Path

# Setup paths to find local utils if needed
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from panoseti_grpc.telemetry.server import serve
    from panoseti_grpc.telemetry.resources import make_rich_logger
except ImportError:
    print("CRITICAL: 'panoseti_grpc' not installed.")
    sys.exit(1)

# --- 1. LOCATE CONFIG ---
# Look for telemetry_config.toml in the SAME directory as this script
LOCAL_CONFIG = Path(__file__).parent / "capture_telemetry_service/telemetry_config.toml"

logger = make_rich_logger("telemetry_daemon")

def main():
    if not LOCAL_CONFIG.exists():
        logger.warning(f"Local config not found at {LOCAL_CONFIG}. Using library defaults.")
        config_arg = None
    else:
        logger.info(f"Using Operational Config: [bold cyan]{LOCAL_CONFIG}[/]", extra={"markup":True})
        config_arg = LOCAL_CONFIG

    # --- 2. LAUNCH ---
    try:
        asyncio.run(serve(
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            grpc_port=int(os.getenv("GRPC_PORT", 50051)),
            config_path=config_arg  # <--- INJECT PATH HERE
        ))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()