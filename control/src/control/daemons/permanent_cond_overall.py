#!/usr/bin/env python3
"""
cond_overall.py

Compute:
  1) Overall Weather Status (local station + NWS sky text + cloud cover)
  2) Overall Conditions per site (Sun + Weather + rain + twilight + freshness)

Disk inputs (as requested):
  Weather dir:
    /mnt/data11/data/palomar/L0/YYYYMMDD/weather/weather.log
    /mnt/data11/data/palomar/L0/YYYYMMDD/weather/palomar_forecast_YYYYMMDD.jsonl
  Dome per site:
    /mnt/data11/data/palomar/L0/YYYYMMDD/<SITE>/dome/roof_status.log

Behavior (as requested):
  - roof OPEN/CLOSED never by itself makes a site BAD (roof is only reported)
  - daytime at Palomar (between sunrise and sunset, local time) => Sun=BAD => site BAD
  - freshness issues produce CAUTION notes but do not ?explain? BAD by themselves
  - Console prints WEATHER + OVERALL CONDITIONS with explicit REASONS
  - Writes JSONL log only (no CSV):
      /mnt/data11/data/palomar/L0/YYYYMMDD/weather/weather_conditions.jsonl
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None  # type: ignore


# ===================== CONFIG =====================
SAVE_ROOT = "/mnt/data11/data/palomar/L0"
SITES_ORDER = ["Gattini", "Winter", "Fern", "PTI"]

INTERVAL_SECONDS = 30
FRESHNESS_MINUTES = 5

# Palomar coordinates
LAT = 33.3563
LON = -116.864

# Palomar local timezone
LOCAL_TZ_NAME = "America/Los_Angeles"


# ===================== TZ =====================
def _tz() -> Any:
    if ZoneInfo is None:
        return UTC
    try:
        return ZoneInfo(LOCAL_TZ_NAME)
    except Exception:
        return UTC


LOCAL_TZ = _tz()


# ===================== HELPERS =====================
def utc_now() -> datetime:
    return datetime.now(UTC)


def ut_yyyymmdd(now_utc: datetime | None = None) -> str:
    n = now_utc or utc_now()
    return n.strftime("%Y%m%d")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def severity_label(sev: int) -> str:
    return ["GOOD", "CAUTION", "BAD"][max(0, min(2, sev))]


def parse_iso_like_to_utc(s: str) -> datetime | None:
    """
    Accepts:
      - "2026-02-06T00:05:19.149"
      - "2026-02-06T00:05:19"
      - "2026-02-06T00:05:19Z"
      - "2026-02-06 00:05:19Z"
      - "... UT"
    Returns aware datetime in UTC.
    """
    if not s or not isinstance(s, str):
        return None

    x = s.strip()
    x = x.replace(" UT", "").replace("UTC", "").strip()
    if " " in x and "T" not in x:
        x = x.replace(" ", "T")

    # If explicit Z
    if x.endswith(("Z", "z")):
        x2 = x[:-1]
        try:
            dt = datetime.fromisoformat(x2)
            return dt.replace(tzinfo=UTC)
        except Exception:
            return None

    # If has offset
    if ("+" in x[10:] or "-" in x[10:]) and len(x) >= 19:
        try:
            dt = datetime.fromisoformat(x)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except Exception:
            return None

    # Otherwise assume UTC
    try:
        dt = datetime.fromisoformat(x)
        return dt.replace(tzinfo=UTC)
    except Exception:
        return None


def parse_time_local(s: str) -> datetime | None:
    """
    Forecast time_local like "YYYY-MM-DD HH:MM" (no tz).
    Treat as LOCAL_TZ.
    """
    if not s or not isinstance(s, str):
        return None
    x = s.strip()
    try:
        dt = datetime.strptime(x, "%Y-%m-%d %H:%M")
        return dt.replace(tzinfo=LOCAL_TZ)
    except Exception:
        try:
            dt = datetime.fromisoformat(x)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=LOCAL_TZ)
            return dt.astimezone(LOCAL_TZ)
        except Exception:
            return None


def is_fresh(dt_utc: datetime | None, max_minutes: int = FRESHNESS_MINUTES) -> bool:
    if dt_utc is None:
        return False
    age = abs((utc_now() - dt_utc).total_seconds())
    return age <= max_minutes * 60


def fmt_num(x: Any, ndp: int = 1) -> str:
    v = safe_float(x)
    if v is None:
        return "?"
    return f"{v:.{ndp}f}"


# ===================== WEATHER LOG READER =====================
# 9 comma-separated fields:
# timestamp, temperature_C, humidity, dew_point_C, barometer_MB,
# wind_speed_MPH, wind_direction, weather_status, last_update
def read_latest_weather_log(weather_log_path: str) -> tuple[dict[str, Any] | None, datetime | None]:
    if not os.path.exists(weather_log_path):
        return None, None

    last = None
    try:
        with open(weather_log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
    except Exception:
        return None, None

    if not last:
        return None, None

    parts = [p.strip() for p in last.split(",")]
    if len(parts) < 9:
        # attempt salvage if last_update contained commas
        if len(parts) >= 8:
            parts = [*parts[:8], ",".join(parts[8:])]
        else:
            return None, None

    ts = parts[0]
    dt_utc = parse_iso_like_to_utc(ts)

    w = {
        "timestamp": ts,
        "temperature_C": safe_float(parts[1]),
        "humidity": safe_float(parts[2]),
        "dew_point_C": safe_float(parts[3]),
        "barometer_MB": safe_float(parts[4]),
        "wind_speed_MPH": safe_float(parts[5]),
        "wind_direction": parts[6] if parts[6] else None,
        "weather_status": parts[7] if parts[7] else None,
        "last_update": parts[8] if parts[8] else None,
    }
    return w, dt_utc


# ===================== FORECAST JSONL READER =====================
def read_latest_forecast_jsonl(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    last = None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
    except Exception:
        return None
    if not last:
        return None
    try:
        return json.loads(last)
    except Exception:
        return None


def pick_best_forecast_hour(fc: dict[str, Any]) -> dict[str, Any] | None:
    hours = fc.get("hours")
    if not isinstance(hours, list) or not hours:
        return None

    now_local = datetime.now(LOCAL_TZ)
    best = None
    best_t = None

    for h in hours:
        if not isinstance(h, dict):
            continue
        tl = h.get("time_local")
        t = parse_time_local(tl) if isinstance(tl, str) else None
        if t is None:
            continue
        if t >= now_local and (best_t is None or t < best_t):
            best = h
            best_t = t

    if best is None:
        best = hours[-1] if isinstance(hours[-1], dict) else None
    return best


# ===================== WEATHER CLASSIFICATION (mirror JS) =====================
def determine_safety_local(w: dict[str, Any]) -> dict[str, Any]:
    """
    Mirror current_status.php JS determineSafety().
    Returns {label, severity, reasons[]}
    """
    t = safe_float(w.get("temperature_C"))
    d = safe_float(w.get("dew_point_C"))
    hum = safe_float(w.get("humidity"))
    wind = safe_float(w.get("wind_speed_MPH"))

    spread = float("inf")
    if t is not None and d is not None:
        spread = t - d

    reasons: list[str] = []
    sev = 0

    # BAD triggers
    if hum is not None and hum > 95:
        reasons.append("Humidity >95%")
        sev = 2
    if spread < 1:
        reasons.append(f"Dewpoint spread <1°C (t?d={spread:.1f}°C)")
        sev = 2
    if wind is not None and wind > 25:
        reasons.append(f"Wind >25 mph ({wind:g})")
        sev = 2
    if t is not None and t < 0:
        reasons.append(f"Frost risk: Temperature <0°C ({t:g}°C)")
        sev = 2

    # CAUTION triggers
    if hum is not None and 80 < hum <= 95:
        reasons.append("Humidity >80%")
        sev = max(sev, 1)
    if 1 <= spread < 3:
        reasons.append(f"Dewpoint spread <3°C (t?d={spread:.1f}°C)")
        sev = max(sev, 1)
    if wind is not None and 15 < wind <= 25:
        reasons.append(f"Wind >15 mph ({wind:g})")
        sev = max(sev, 1)

    if not reasons:
        return {"label": "GOOD", "severity": 0, "reasons": ["Conditions nominal"]}

    return {"label": severity_label(sev), "severity": sev, "reasons": reasons}


def classify_nws_text(forecast_text: str | None) -> str:
    # Mirror JS classifyNWS()
    if not forecast_text:
        return "CAUTION"
    s = str(forecast_text).lower()
    if ("clear" in s) or ("sunny" in s):
        return "GOOD"
    if ("partly" in s) or ("mostly" in s):
        return "CAUTION"
    return "BAD"


def classify_cloud_cover(cloud_pc: Any) -> str:
    # Mirror JS classifyCloud()
    c = safe_float(cloud_pc)
    if c is None:
        return "CAUTION"
    if c <= 20:
        return "GOOD"
    if c <= 50:
        return "CAUTION"
    return "BAD"


def combine_overall_weather(local_label: str, nws_label: str, cloud_label: str) -> dict[str, Any]:
    overall = "GOOD"
    if "BAD" in (local_label, nws_label, cloud_label):
        overall = "BAD"
    elif "CAUTION" in (local_label, nws_label, cloud_label):
        overall = "CAUTION"

    sev = {"GOOD": 0, "CAUTION": 1, "BAD": 2}[overall]
    return {
        "label": overall,
        "severity": sev,
        "reasons": [
            f"Local station: {local_label}",
            f"NWS sky: {nws_label}",
            f"Cloud cover: {cloud_label}",
        ],
    }


# ===================== SUN: hard daytime BAD =====================
def sun_times_for_date_local(date_local: datetime, lat: float = LAT, lon: float = LON) -> tuple[datetime, datetime]:
    """
    NOAA-style approximate sunrise/sunset in LOCAL_TZ for the given local date.
    """
    d = date_local.astimezone(LOCAL_TZ)
    n = int(d.strftime("%j"))  # day of year

    B = math.radians((360 / 364) * (n - 81))
    eot = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)  # minutes
    decl = math.radians(23.45) * math.sin(math.radians((360 / 365) * (284 + n)))

    lat_r = math.radians(lat)
    cos_h0 = (math.sin(math.radians(-0.83)) - math.sin(lat_r) * math.sin(decl)) / (math.cos(lat_r) * math.cos(decl))
    cos_h0 = max(-1.0, min(1.0, cos_h0))
    h0 = math.acos(cos_h0)

    tz_offset = d.utcoffset()
    tz_offset_h = tz_offset.total_seconds() / 3600.0 if tz_offset is not None else 0.0
    solar_noon_local = 12 + tz_offset_h - lon / 15.0 - eot / 60.0
    day_len_h = 2 * math.degrees(h0) / 15.0

    def hours_to_local(hh: float) -> datetime:
        base = d.replace(hour=0, minute=0, second=0, microsecond=0)
        return base + timedelta(hours=hh)

    sunrise = hours_to_local(solar_noon_local - day_len_h / 2)
    sunset = hours_to_local(solar_noon_local + day_len_h / 2)
    return sunrise, sunset


def sun_phase_now() -> dict[str, Any]:
    """
    HARD RULE: between sunrise and sunset (local) => BAD, else GOOD.
    """
    now = datetime.now(LOCAL_TZ)
    sunrise, sunset = sun_times_for_date_local(now)

    if sunrise <= now <= sunset:
        phase = "BAD"
        reason = "Daytime (between sunrise and sunset)"
    else:
        phase = "GOOD"
        reason = "Nighttime (sunset ? sunrise)"

    return {
        "phase": phase,
        "reason": reason,
        "sunrise_local": sunrise.isoformat(timespec="seconds"),
        "sunset_local": sunset.isoformat(timespec="seconds"),
    }


# ===================== DOME LOG READER =====================
def read_last_json_line(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    last = None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
    except Exception:
        return None
    if not last:
        return None
    try:
        return json.loads(last)
    except Exception:
        return None


def dome_snapshot_for_site(yyyymmdd: str, site: str) -> tuple[dict[str, Any] | None, datetime | None, str]:
    p = os.path.join(SAVE_ROOT, yyyymmdd, site, "dome", "roof_status.log")
    obj = read_last_json_line(p)
    if not obj:
        return None, None, p
    dt = parse_iso_like_to_utc(str(obj.get("timestamp", "")))
    return obj, dt, p


def sensor_state_str(flag: Any) -> str:
    if flag in ("true", "1", 1, True):
        return "UNSAFE"
    if flag in ("false", "0", 0, False):
        return "SAFE"
    return "UNKNOWN"


def sensor_severity(flag: Any, missing_label: str, issues: list[str]) -> int:
    """
    Mirror JS sensorSeverity():
      true  -> BAD (2)
      false -> GOOD (0)
      unknown/missing -> CAUTION (1) + issues
    """
    if flag in ("true", "1", 1, True):
        return 2
    if flag in ("false", "0", 0, False):
        return 0
    issues.append(missing_label)
    return 1


# ===================== OVERALL PER SITE =====================
def overall_for_site(
    site: str,
    dome_obj: dict[str, Any] | None,
    dome_dt: datetime | None,
    overall_weather_label: str,
    overall_weather_sev: int,
    weather_fresh: bool,
    dome_fresh: bool,
    sp: dict[str, Any],
) -> dict[str, Any]:
    """
    Returns a dict with:
      label/severity (final)
      reasons[]  -> ALWAYS populated with the actual drivers (like PHP panel text)
      issues[]   -> ONLY data-quality / uncertainty notes
    """
    issues: list[str] = []
    reasons: list[str] = []

    worst = 0

    # --- Freshness / data-quality ---
    if not dome_fresh:
        issues.append("Stale dome data")
        worst = max(worst, 1)

    if not weather_fresh:
        issues.append("Stale weather data")
        issues.append("Ignoring weather (stale)")
        # stale weather in PHP becomes UNKNOWN/CAUTION contribution
        wx_label = "UNKNOWN"
        wx_sev = 1
    else:
        wx_label = overall_weather_label
        wx_sev = overall_weather_sev

    # --- Dome / sensors ---
    roof_pos = "UNKNOWN"
    rain_sev = 1
    twi_sev = 1
    rain_state = "UNKNOWN"
    twi_state = "UNKNOWN"

    if dome_obj and isinstance(dome_obj, dict):
        roof_pos = str((dome_obj.get("roofSignalsStatus") or {}).get("roofPosition") or "UNKNOWN").upper()

        if site.lower() == "gattini":
            # Not installed => treat GOOD (0) and explain explicitly
            rain_sev = 0
            twi_sev = 0
            rain_state = "NOT INSTALLED"
            twi_state = "NOT INSTALLED"
        else:
            rain_obj = dome_obj.get("rainSensorStatus") or {}
            twi_obj = dome_obj.get("twilightSwitchStatus") or {}
            rain_flag = rain_obj.get("unsafeCondition")
            twi_flag = twi_obj.get("unsafeCondition")

            rain_state = sensor_state_str(rain_flag)
            twi_state = sensor_state_str(twi_flag)

            rain_sev = sensor_severity(rain_flag, "Rain sensor missing", issues)
            twi_sev = sensor_severity(twi_flag, "Twilight sensor missing", issues)
    else:
        issues.append("No dome data available")
        worst = max(worst, 1)
        # keep sensor states UNKNOWN

    worst = max(worst, rain_sev, twi_sev)

    # --- Weather contribution (fresh => BAD/CAUTION/GOOD; stale => UNKNOWN/CAUTION) ---
    worst = max(worst, wx_sev)

    # --- Sun ALWAYS applies (your rule) ---
    sun_sev = 2 if sp["phase"] == "BAD" else 0
    worst = max(worst, sun_sev)

    # --- Build always-on reasons (what *drove* the decision) ---
    # (This is why your previous "issues: none" was confusing.)
    reasons.append(f"Sun phase: {sp['phase']} ? {sp['reason']}")
    reasons.append(f"Overall Weather: {wx_label}")

    # Sensors: explicit states (like the PHP modal explanation)
    reasons.append(f"Rain sensor: {rain_state}")
    reasons.append(f"Twilight sensor: {twi_state}")

    # Freshness notes (kept separate but also visible if present)
    if issues:
        for x in issues:
            reasons.append(f"NOTE: {x}")

    label = severity_label(worst)

    return {
        "label": label,
        "severity": worst,
        "reasons": reasons,
        "issues": issues,
        "roof_position": roof_pos,                 # reported only, not used as driver
        "dome_timestamp": dome_obj.get("timestamp") if dome_obj else None,
        "sun_phase": sp["phase"],
        "sun_reason": sp["reason"],
        "sunrise_local": sp["sunrise_local"],
        "sunset_local": sp["sunset_local"],
        "weather": wx_label,
        "rain": rain_state,
        "twilight": twi_state,
    }


# ===================== JSONL WRITER =====================
def append_jsonl(path: str, obj: dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ===================== MAIN LOOP =====================
def main() -> int:
    print(f"[INFO] Weather dir: {SAVE_ROOT}/YYYYMMDD/weather")
    print(f"[INFO] Dome per-site: {SAVE_ROOT}/YYYYMMDD/<SITE>/dome/roof_status.log")
    print(f"[INFO] Interval: {INTERVAL_SECONDS}s  Freshness: {FRESHNESS_MINUTES} min")

    while True:
        now_utc = utc_now()
        yyyymmdd = ut_yyyymmdd(now_utc)

        weather_dir = os.path.join(SAVE_ROOT, yyyymmdd, "weather")
        weather_log = os.path.join(weather_dir, "weather.log")
        forecast_jsonl = os.path.join(weather_dir, f"palomar_forecast_{yyyymmdd}.jsonl")

        # ---- Read weather.log ----
        w_latest, w_dt = read_latest_weather_log(weather_log)
        weather_fresh = is_fresh(w_dt, FRESHNESS_MINUTES) if w_dt else False

        local_safety = (
            determine_safety_local(w_latest)
            if w_latest else {"label": "CAUTION", "severity": 1, "reasons": ["No weather data"]}
        )

        # ---- Read forecast jsonl ----
        fc = read_latest_forecast_jsonl(forecast_jsonl)
        best_hour = pick_best_forecast_hour(fc) if fc else None

        if best_hour:
            nws_text = best_hour.get("forecast")
            nws_status = classify_nws_text(nws_text if isinstance(nws_text, str) else None)
            cloud_status = classify_cloud_cover(best_hour.get("cloud_cover_pc"))
            best_time_local = best_hour.get("time_local")
        else:
            nws_status = "CAUTION"
            cloud_status = "CAUTION"
            best_time_local = None

        overall_weather = combine_overall_weather(str(local_safety["label"]), str(nws_status), str(cloud_status))

        # ---- Sun phase ----
        sp = sun_phase_now()

        # ---- Dome logs ----
        dome_paths: dict[str, str] = {}
        dome_objs: dict[str, dict[str, Any] | None] = {}
        dome_dts: dict[str, datetime | None] = {}
        dome_fresh_map: dict[str, bool] = {}

        for site in SITES_ORDER:
            obj, dt, path = dome_snapshot_for_site(yyyymmdd, site)
            dome_paths[site] = path
            dome_objs[site] = obj
            dome_dts[site] = dt
            dome_fresh_map[site] = is_fresh(dt, FRESHNESS_MINUTES) if dt else False

        # ---- Per-site overall ----
        sites_overall: dict[str, Any] = {}
        for site in SITES_ORDER:
            ov = overall_for_site(
                site=site,
                dome_obj=dome_objs[site],
                dome_dt=dome_dts[site],
                overall_weather_label=overall_weather["label"],
                overall_weather_sev=overall_weather["severity"],
                weather_fresh=weather_fresh,
                dome_fresh=dome_fresh_map[site],
                sp=sp,
            )
            sites_overall[site] = ov

        # ===================== CONSOLE PRINT =====================
        print("")
        print(f"Weather: local={local_safety['label']} nws={nws_status} cloud={cloud_status} => overall={overall_weather['label']} (fresh={weather_fresh})")
        if w_latest:
            print(
                f"  Temp={fmt_num(w_latest.get('temperature_C'))}°C  "
                f"RH={fmt_num(w_latest.get('humidity'))}%  "
                f"Dew={fmt_num(w_latest.get('dew_point_C'))}°C  "
                f"Wind={fmt_num(w_latest.get('wind_speed_MPH'))} mph"
            )
        else:
            print("  (no weather.log data)")

        print("Overall conditions:")
        for site in SITES_ORDER:
            ov = sites_overall[site]
            roof = ov.get("roof_position", "UNKNOWN")
            # Keep the compact line, but now show first-class reasons
            print(f"  {site:<7} {ov['label']:<7} roof={roof:<7} Sun:{ov.get('sun_phase')}, Wx:{ov.get('weather')}")
            for r in ov.get("reasons", []):
                print(f"    - {r}")

        # ===================== JSONL LOG (no CSV) =====================
        out_jsonl = os.path.join(weather_dir, "weather_conditions.jsonl")
        snapshot = {
            "utc_timestamp": now_utc.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "inputs": {
                "weather_log": weather_log,
                "forecast_jsonl": forecast_jsonl,
                "dome_roof_status_logs": dome_paths,
            },
            "weather_latest": w_latest,
            "forecast_best_hour": {
                "time_local": best_time_local,
                "forecast": best_hour.get("forecast") if best_hour else None,
                "cloud_cover_pc": best_hour.get("cloud_cover_pc") if best_hour else None,
                "PoP_pc": best_hour.get("PoP_pc") if best_hour else None,
            } if best_hour else None,
            "computed": {
                "weather_fresh": weather_fresh,
                "local_safety": local_safety,
                "nws_status": nws_status,
                "cloud_status": cloud_status,
                "overall_weather": overall_weather,
                "sun_phase_now": sp,
                "sites_overall": sites_overall,
            },
        }
        append_jsonl(out_jsonl, snapshot)

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.\n")
        raise



