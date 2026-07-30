#! /usr/bin/env python3

"""
Script for capturing metadata from each ethernet outlet
and storing it in the Redis database.
"""


import time
from typing import Any

import control.power as power
from control.utils import config_file, redis_utils
from control.utils.pydantic_config_models import WpsConfig

# Time between updates.
UPDATE_INTERVAL = 1


def get_wps_fields(wps_model: WpsConfig | dict[str, Any]) -> dict[str, Any]:
    """Retrieve power status and timestamp for a specific WPS unit.

    Args:
        wps_model: Configuration model or dict for the target WPS unit.

    Returns:
        A dictionary of fields to write into Redis ('Computer_UTC', 'POWER').

    Raises:
        Exception: If the WPS unit cannot be queried.
    """
    try:
        power_status = "ON" if power.quabo_power_query(wps_model) else "OFF"
    except Exception:
        print(f'capture_power.py: Failed to query {wps_model}. The login info for this UPS may be incorrect."')
        raise
    rkey_fields: dict[str, Any] = {
        'Computer_UTC': time.time(),
        'POWER': power_status
    }
    return rkey_fields


def get_wps_rkey(wps_key: str) -> str:
    """Determine the Redis key for a named WPS unit.

    Args:
        wps_key: The configuration key for the WPS unit (e.g., 'wps1').

    Returns:
        The corresponding Redis key as an uppercase string.
    """
    return wps_key.upper()


def main() -> None:
    """Background loop that periodically snapshots WPS power states into Redis."""
    r = redis_utils.redis_init()
    obs_config = config_file.get_obs_config()
    
    extra = obs_config.model_extra or {}
    wps_keys = [key for key in extra if 'wps' in key.lower()]
    
    print("capture_power.py: Running...")
    while True:
        for wps_key in wps_keys:
            rkey = get_wps_rkey(wps_key)
            wps_data = extra[wps_key]
            # Convert to model if it's a dict
            wps_model = WpsConfig(**wps_data) if isinstance(wps_data, dict) else wps_data
            fields = get_wps_fields(wps_model)
            redis_utils.store_in_redis(r, rkey, fields)
        time.sleep(UPDATE_INTERVAL)


if __name__ == "__main__":
    main()

