#!/usr/bin/env python3
import time
import requests
import redis
import argparse
import datetime
from astropy.time import Time


URL = "http://10.200.130.100/admin/scripts/getMainWeather.php"
REDIS_KEY = "WEATHER"
REDIS_HOST = "localhost"
REDIS_PORT = 6379


def deg_to_dir(deg):
    """Convert numeric wind direction (degrees) to compass label."""
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    try:
        i = int((deg + 11.25) / 22.5) % 16
        return directions[i]
    except Exception:
        return "N/A"


def get_weather():
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
    def safe_float(x):
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
        "barometer_MB": w.get("pressure", "N/A"),
        "wind_speed_MPH": round(windspd_mph, 1) if windspd_mph is not None else "N/A",
        "wind_direction": winddir,
        "weather_status": w.get("wstatus", "N/A"),
        "last_update": w.get("date", "N/A")
    }

    return weather


def save_to_redis(weather):
    """Store weather hash in Redis."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.hset(REDIS_KEY, mapping=weather)
        print(f"? Updated Redis key '{REDIS_KEY}'")
    except Exception as e:
        print(f"? Redis error: {e}")


def main(interval):
    print(f"Starting Palomar weather capture every {interval}s...")
    while True:
        weather = get_weather()
        if weather:
            print("\n=== Palomar Observatory ? WINTER Weather ===")
            for k, v in weather.items():
                print(f"{k:<18}: {v}")
            save_to_redis(weather)
        else:
            print("?? No weather data available.")

        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture Palomar WINTER weather and store in Redis.")
    parser.add_argument("--interval", "-i", type=int, default=30,
                        help="Polling interval in seconds (default: 30)")
    args = parser.parse_args()

    main(args.interval)
