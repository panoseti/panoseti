#!/usr/bin/env python3

import contextlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

# ============================================================
# === Python version compatibility for type hints (3.7?3.12) ==
# ============================================================
StrOrNone = str | None
IntOrNone = int | None

# =====================
# ====== CONFIG =======
# =====================
LAT, LON = 33.35, -116.87               # Palomar Mountain
HOURS_AHEAD = 96                        # 4 days

WINDOW_NEAR_HOURS = 24                  # now ? +24h
WINDOW_FAR_START  = 24                  # +24h ? +96h
WINDOW_FAR_END    = 96

# === Thresholds ===
POP_CAUTION = 20          # %
WIND_CAUTION = 15         # mph
GUST_ALERT = 25           # mph
CLOUD_CAUTION = 70        # % cloud cover caution threshold
TEMP_FROST_C = 0          # °C frost risk
TEMP_HOT_C   = 30         # °C heat caution threshold   <-- Add this


INTERVAL_SECONDS = 30 * 60              # 30 min loop

# Daily archives are under UTC date inside L0
BASE_DIR = Path("/mnt/data11/data/palomar/L0")

# Per-run snapshots (before SCP)
SNAP_JSON_NAME = "palomar_forecast_current.json"
SNAP_LOG_NAME  = "alerts_current.log"

# Remote (cylon)
BANDWIDTH_LIMIT = 40000
REMOTE_SERVER = "panoseti@132.239.146.24"
REMOTE_WEBCAM_DIR  = "/web/panoseti-palomar/current"
REMOTE_WEBCAM_DIR2 = "/web/panoseti-palomar/current"

UA = {"User-Agent": "PANOSETI-ops (contact: ops@example.com)"}

# =====================
# ===== UTILITIES =====
# =====================
def now_utc() -> datetime:
    return datetime.now(UTC)

def fmt_local(ts: datetime) -> str:
    return ts.astimezone().strftime("%Y-%m-%d %H:%M")

def fmt_utc(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%SZ")

def parse_iso_any(s: str) -> datetime:
    if s.endswith("Z"):
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)

def mph_from_str(s: StrOrNone) -> IntOrNone:
    if not s:
        return None
    parts = s.replace("to", " ").split()
    nums = []
    for p in parts:
        with contextlib.suppress(Exception):
            nums.append(int(p))
    return max(nums) if nums else None

def time_left_str(future: datetime, ref: datetime) -> str:
    delta = future - ref
    total_min = max(0, int(delta.total_seconds() // 60))
    h, m = divmod(total_min, 60)
    return f"in {h}h {m}m" if h else f"in {m}m"

def run_cmd(cmd: str) -> bool:
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"[CMD FAIL] {cmd}\n{e}\n")
        return False

# ==========================
# ====== DATA FETCH ========
# ==========================
def get_json(url: str) -> Any:
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_nws_endpoints(lat: float, lon: float) -> dict[str, Any]:
    meta = get_json(f"https://api.weather.gov/points/{lat},{lon}")
    props = meta["properties"]
    return {
        "hourly_url": props["forecastHourly"],
        "zone": props.get("forecastZone") or props.get("county")
    }

def fetch_nws_hourly(hourly_url: str) -> list[dict[str, Any]]:
    return get_json(hourly_url)["properties"]["periods"]

def fetch_openmeteo_cloudcover(lat: float, lon: float, days: int = 4) -> dict[str, int]:
    data = get_json(
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly=cloudcover&timezone=UTC&forecast_days={days}"
    )
    times = data.get("hourly", {}).get("time", []) or []
    cover = data.get("hourly", {}).get("cloudcover", []) or []
    cc = {}
    for t, c in zip(times, cover, strict=False):
        t_utc = parse_iso_any(t)
        key = t_utc.strftime("%Y-%m-%dT%H:00Z")
        cc[key] = int(c)
    return cc

def fetch_active_alerts(zone_url: StrOrNone) -> list[dict[str, Any]]:
    if not zone_url:
        return []
    zone_code = zone_url.split("/")[-1]
    alerts_api = f"https://api.weather.gov/alerts/active?zone={zone_code}"
    data = get_json(alerts_api)
    out = []
    for feat in data.get("features", []):
        p = feat.get("properties", {})
        out.append({
            "event": p.get("event"),
            "severity": p.get("severity"),
            "headline": p.get("headline"),
            "effective": p.get("effective"),
            "expires": p.get("expires"),
            "instruction": p.get("instruction")
        })
    return out

# ==========================
# ====== PROCESSING ========
# ==========================
def build_rows(periods: list[dict[str, Any]], cloud_map: dict[str, int]) -> list[dict[str, Any]]:
    now = now_utc()
    cutoff = now + timedelta(hours=HOURS_AHEAD)
    rows = []
    for p in periods:
        start = parse_iso_any(p["startTime"])
        if start > cutoff:
            continue

        temp = p.get("temperature")
        wind_s = p.get("windSpeed") or ""
        gust_s = p.get("windGust") or ""
        pop = p.get("probabilityOfPrecipitation", {}).get("value")
        short = p.get("shortForecast") or ""

        w_mph = mph_from_str(wind_s)
        g_mph = mph_from_str(gust_s)

        key = start.strftime("%Y-%m-%dT%H:00Z")
        cc = cloud_map.get(key)

                # Convert temperature to °C
        (temp - 32) * 5.0 / 9.0 if temp is not None else None

        rows.append({
            "time_utc": fmt_utc(start),
            "time_local": fmt_local(start),
            "temp_F": temp,
            "temp_C": (temp-32)*5/9 if temp is not None else None,    # ? added Celsius
            "PoP_pc": (None if pop is None else int(pop)),
            "wind_str": wind_s,
            "gust_str": gust_s,
            "wind_mph": w_mph,
            "gust_mph": g_mph,
            "cloud_cover_pc": cc,
            "forecast": short
        })

    return rows

# expose thresholds for the alert filter logic
POP_CAUTION = POP_CAUTION        # precipitation %
WIND_CAUTION = WIND_CAUTION      # wind mph steady caution
GUST_ALERT = GUST_ALERT          # gust mph alert
CLOUD_CAUTION = CLOUD_CAUTION if 'CLOUD_CAUTION' in globals() else 70

def flag_window(rows: list[dict[str, Any]], h_start: float, h_end: float, first_only: bool = False) -> list[str]:
    now = now_utc()
    results: list[Any] = []

    for r in rows:
        t = parse_iso_any(r["time_utc"])
        dh = (t - now).total_seconds() / 3600.0
        if not (h_start <= dh < h_end):
            continue

        events = []

        # PoP
        if r["PoP_pc"] is not None and r["PoP_pc"] >= POP_CAUTION:
            events.append(("PoP", f"PoP?{POP_CAUTION}% ({r['PoP_pc']}%)"))

        # Wind
        if r["wind_mph"] is not None and r["wind_mph"] > WIND_CAUTION:
            events.append(("Wind", f"Wind>{WIND_CAUTION} mph ({r['wind_mph']})"))

        # Gust
        if r["gust_mph"] is not None and r["gust_mph"] > GUST_ALERT:
            events.append(("Gust", f"Gust>{GUST_ALERT} mph ({r['gust_mph']})"))

        # Cloud
        if r["cloud_cover_pc"] is not None and r["cloud_cover_pc"] >= CLOUD_CAUTION:
            events.append(("Cloud", f"Cloud>{CLOUD_CAUTION}% ({r['cloud_cover_pc']}%)"))

        # Temp < 0°C (frost)
        if r["temp_C"] is not None and r["temp_C"] < TEMP_FROST_C:
            events.append(("Temp", f"Frost risk: Temp <0°C ({r['temp_C']:.1f}°C)"))

        # Temp > 30°C (hot)
        if r["temp_C"] is not None and r["temp_C"] >= TEMP_HOT_C:
            events.append(("Hot", f"High temperature: Temp ?{TEMP_HOT_C}°C ({r['temp_C']:.1f}°C)"))


        for key, msg in events:
            entry = f"{r['time_local']} ? {msg}  [{time_left_str(t, now)}]"

            if first_only:   # return only first event of each category
                if key not in {k for k, _ in results}:
                    results.append((key, entry))
            else:
                results.append((key, entry))

    return [msg for _, msg in results]


# ======================
# ====== STORAGE =======
# ======================
def one_cycle(tmp_dir: Path) -> tuple[Path, Path]:
    endpoints = fetch_nws_endpoints(LAT, LON)
    periods = fetch_nws_hourly(endpoints["hourly_url"])
    cloud_map = fetch_openmeteo_cloudcover(LAT, LON, days=4)
    alerts = fetch_active_alerts(endpoints.get("zone"))

    rows = build_rows(periods, cloud_map)

    now = now_utc()
    d_utc = now.strftime("%Y%m%d")
    day_dir = BASE_DIR / d_utc / "weather"
    day_dir.mkdir(parents=True, exist_ok=True)

    gen_local = fmt_local(now)
    gen_utc = fmt_utc(now)

    snap_json_obj = {
        "generated_at_utc": gen_utc,
        "generated_at_local": gen_local,
        "location": {"lat": LAT, "lon": LON},
        "thresholds": {
            "PoP_caution_pc": POP_CAUTION,
            "cloud_caution_pc": CLOUD_CAUTION,
            "wind_caution_mph": WIND_CAUTION,
            "gust_alert_mph": GUST_ALERT,
            "temp_frost_c": TEMP_FROST_C,
            "temp_hot_c": TEMP_HOT_C 
        },
        "hours": rows,
        "active_nws_alerts": alerts
    }

    # Build log (human-readable; keep % characters)
    header = (
        f"[{gen_local} | {gen_utc}]\n"
        f"Palomar weather alert report\n"
        f"Thresholds: PoP?{POP_CAUTION}%, Cloud?{CLOUD_CAUTION}%, Wind>{WIND_CAUTION} mph, Gust>{GUST_ALERT} mph, Temp<{TEMP_FROST_C}°C (frost), Temp?{TEMP_HOT_C}°C (heat)\n"

    )
    header = (
        f" Weather Forecast & Alert Reports (Palomar)\n"
        f" Generated Local: {gen_local} (UT:{gen_utc})\n"
        f" Panoseti Alert Thresholds:\n"
        f" Cloud: {CLOUD_CAUTION}%, Precipitation Probability: {POP_CAUTION}%, Wind > {WIND_CAUTION} mph, Gust > {GUST_ALERT} mph, Frost Risk: Temp < {TEMP_FROST_C}°C, Heat Caution: Temp > {TEMP_HOT_C}°C\n"
        f"---------------------------------------------\n"
    )


    near_lines = flag_window(rows, 0, WINDOW_NEAR_HOURS)
    far_next   = flag_window(rows, WINDOW_FAR_START, WINDOW_FAR_END, first_only=True)


    
    if alerts:
        alert_block = (
            ">>> Active NWS Hazard Alerts for Palomar <<<\n" +
            "\n".join(
                f"> {a.get('event')} ({a.get('severity','')}) ? {a.get('headline','')}"
            for a in alerts
        ) +
        "\n"
        )
    else:
        alert_block = (
            ">>> No active NWS hazard alerts for the Palomar forecast zone.\n"
        )


    if near_lines:
        near_block = "=== UPCOMING WINDOWS (next 24h) ===\n" + "\n".join(f"? {ln}" for ln in near_lines)
    else:
        near_block = "No PoP/Wind/Cloud thresholds triggered in next 24h."

    if far_next:
        far_block = "=== NEXT TRIGGERS (1?4 days) ===\n" + "\n".join(f"? {ln}" for ln in far_next)
    else:
        far_block = "No significant PoP/Wind/Cloud/Temp events in next 1?4 days."

        # ---- NEXT 24 HOURS ----
    if near_lines:
        near_block = (
            ">>> Conditions Expected Within the Next 24 Hours (Local Time) <<<\n"
            + "\n".join(f"> {ln}" for ln in near_lines)
        )
    else:
        near_block = (
            ">>> No threshold-triggering cloud / wind / precipitation / temperature conditions expected within the next 24 hours.\n"
        )

        # ---- NEXT 1?4 DAYS ----
    if far_next:
        far_block = (
            ">>> Next Significant Condition Between 24 Hours and 4 Days <<<\n"
            + "\n".join(f"> {ln}" for ln in far_next)
        )
    else:
        far_block = (
            ">>> No significant threshold events forecast in the 1?4 day range.\n"
        )


    snap_log_text = f"{header}\n{alert_block}\n\n{near_block}\n\n{far_block}\n"

    # Paths
    latest_json = tmp_dir / SNAP_JSON_NAME
    latest_log  = tmp_dir / SNAP_LOG_NAME

    latest_json.parent.mkdir(parents=True, exist_ok=True)
    with open(latest_json, "w") as f:
        json.dump(snap_json_obj, f, indent=2)
    with open(latest_log, "w") as f:
        f.write(snap_log_text)

    # Append to daily rolling files
    daily_jsonl = day_dir / f"palomar_forecast_{d_utc}.jsonl"
    with open(daily_jsonl, "a") as f:
        f.write(json.dumps(snap_json_obj) + "\n")

    daily_log = day_dir / f"alerts_{d_utc}.log"
    with open(daily_log, "a") as f:
        f.write("\n" + snap_log_text + "\n")

    return latest_json, latest_log

def scp_latest(json_path: Path, log_path: Path) -> None:
    run_cmd(f"scp -l {BANDWIDTH_LIMIT} {json_path} {REMOTE_SERVER}:{REMOTE_WEBCAM_DIR}/{SNAP_JSON_NAME}")
    run_cmd(f"scp -l {BANDWIDTH_LIMIT} {log_path} {REMOTE_SERVER}:{REMOTE_WEBCAM_DIR}/{SNAP_LOG_NAME}")
    run_cmd(f'ssh {REMOTE_SERVER} "chmod 664 {REMOTE_WEBCAM_DIR2}/{SNAP_JSON_NAME}"')
    run_cmd(f'ssh {REMOTE_SERVER} "chmod 664 {REMOTE_WEBCAM_DIR2}/{SNAP_LOG_NAME}"')

# ====================
# ===== MAIN LOOP ====
# ====================
def main() -> None:
    tmp_dir = Path("./palomar_weather_tmp")
    tmp_dir.mkdir(exist_ok=True)
    while True:
        try:
            latest_json, latest_log = one_cycle(tmp_dir)
            scp_latest(latest_json, latest_log)
        except Exception as e:
            sys.stderr.write(f"[ERROR] {e}\n")
        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
