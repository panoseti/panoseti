#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import time
from typing import Any

import redis
import requests
from astropy.time import Time

URL = "http://10.200.130.100/admin/scripts/getMainWeather.php"
REDIS_KEY = "WEATHER"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
BASE_DIR = "/mnt/data11/data/palomar/L0"  # Root data directory
# Remote web syn)
REMOTE_SERVER = "panoseti@132.239.146.24"
REMOTE_WEATHER_DIR = "/web/panoseti-palomar/current"
REMOTE_WEATHER_DIR2 = "/web/panoseti-palomar/current"
BANDWIDTH_LIMIT = 40000  # kbit/s


def deg_to_dir(deg: float) -> str:
    """Convert numeric wind direction (degrees) to compass label."""
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    try:
        i = int((deg + 11.25) / 22.5) % 16
        return directions[i]
    except Exception:
        return "N/A"


def get_weather() -> dict[str, Any] | None:
    """Fetch current WINTER weather data from Palomar server."""
    try:
        r = requests.get(URL, timeout=5)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"? Error fetching weather data: {e}")
        return None

    if not data or "winter" not in data:
        print("?? No 'winter' data found in JSON.")
        return None

    w = data["winter"]

    # Parse values safely
    def safe_float(x: Any) -> float | None:
        try:
            return float(x)
        except Exception:
            return None

    tempC = safe_float(w.get("outat"))
    dewC = safe_float(w.get("outdp"))
    winddir_deg = safe_float(w.get("wdir"))
    windspd_ms = safe_float(w.get("wspeed"))
    winddir = deg_to_dir(winddir_deg if winddir_deg is not None else 0)
    windspd_mph = windspd_ms * 2.23694 if windspd_ms is not None else None
    now = Time.now()
    iso = now.isot

    weather = {
        "timestamp": iso,
        "Computer_UTC": time.time(),
        "temperature_C": round(tempC, 1) if tempC is not None else "N/A",
        "temperature_F": round(tempC * 1.8 + 32, 1) if tempC is not None else "N/A",
        "humidity": w.get("outrh", "N/A"),
        "dew_point_C": round(dewC, 1) if dewC is not None else "N/A",
        "barometer_MB": float(w.get("pressure", "nan")) if w.get("pressure") not in (None, "") else float("nan"),
        "wind_speed_MPH": round(windspd_mph, 1) if windspd_mph is not None else "N/A",
        "wind_direction": winddir,
        "weather_status": w.get("wstatus", "N/A"),
        "last_update": w.get("date", "N/A"),
    }

    return weather


def save_to_redis(weather: dict[str, Any]) -> None:
    """Store weather hash in Redis."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.hset(REDIS_KEY, mapping=weather)
        print(f"? Updated Redis key '{REDIS_KEY}'")
    except Exception as e:
        print(f"? Redis error: {e}")


def write_log(weather: dict[str, Any]) -> None:
    """Append weather data to a daily log file with header if new."""
    # Create directory /mnt/data11/data/palomar/L0/YYYYMMDD/weather
    utc_date = datetime.datetime.now(datetime.UTC).replace(tzinfo=None).strftime("%Y%m%d")
    log_dir = os.path.join(BASE_DIR, utc_date, "weather")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "weather.log")

    # Define field names and order
    headers = [
        "timestamp",
        "temperature_C",
        "humidity",
        "dew_point_C",
        "barometer_MB",
        "wind_speed_MPH",
        "wind_direction",
        "weather_status",
        "last_update",
    ]

    # Write header if file does not exist or is empty
    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        with open(log_file, "w") as f:
            f.write(",".join(headers) + "\n")

    # Build one-line CSV log entry
    fields = [
        weather.get("timestamp", ""),
        f"{weather.get('temperature_C', '')}",
        f"{weather.get('humidity', '')}",
        f"{weather.get('dew_point_C', '')}",
        f"{weather.get('barometer_MB', '')}",
        f"{weather.get('wind_speed_MPH', '')}",
        f"{weather.get('wind_direction', '')}",
        f"{weather.get('weather_status', '')}",
        f"{weather.get('last_update', '')}",
    ]
    line = ",".join(map(str, fields)) + "\n"

    # Append data line
    with open(log_file, "a") as f:
        f.write(line)

def copy_weather_to_cylon(weather: dict[str, Any]) -> None:
    """Write latest weather to weather_current.json and transfer to cylon."""
    tmp = "/tmp/weather_current.json"
    try:
        os.makedirs("/tmp", exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(weather, f, indent=2)

        # Upload file (same as dome sync)
        os.system(f"scp -l {BANDWIDTH_LIMIT} {tmp} {REMOTE_SERVER}:{REMOTE_WEATHER_DIR}/")

        # Fix permissions for web download
        os.system(f'ssh {REMOTE_SERVER} "chmod 644 {REMOTE_WEATHER_DIR2}/weather_current.json || true"')

        print("? Updated weather_current.json on cylon")

    except Exception as e:
        print(f"? Failed to upload weather_current.json: {e}")


def main(interval: int) -> None:
    print(f"Starting Palomar weather capture every {interval}s...")
    while True:
        weather = get_weather()
        if weather:
            print("\n=== Palomar Observatory ? WINTER Weather ===")
            for k, v in weather.items():
                print(f"{k:<18}: {v}")
            save_to_redis(weather)
            write_log(weather)
            copy_weather_to_cylon(weather)
        else:
            print("?? No weather data available.")

        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture Palomar WINTER weather and store in Redis + log file.")
    parser.add_argument("--interval", "-i", type=int, default=30,
                        help="Polling interval in seconds (default: 30)")
    args = parser.parse_args()

    main(args.interval)
