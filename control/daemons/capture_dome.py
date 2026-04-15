#!/usr/bin/env python3
import datetime
import json
import os
import subprocess
import time

import requests

# ================== CONFIG ==================
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "capture_dome", "sites_config.json")
SAVE_ROOT = "/mnt/data11/data/palomar/L0"
PORT = 8081
INTERVAL_SECONDS = 30

# Remote / Web sync config (UNCHANGED CONNECTION STYLE)
BANDWIDTH_LIMIT = 40000   # kbit/s
REMOTE_SERVER = "panoseti@132.239.146.24"
REMOTE_DOME_DIR = "/web/panoseti-palomar/current"   # **SINGLE FIXED PATH**
REMOTE_DOME_DIR2 = "/web/panoseti-palomar/current"  # used for chmod

ENDPOINTS = {
    "roofSignalsStatus": "/rest/roofSignalsStatus",
    "roofControlSettings": "/rest/roofControlSettings",
    "motorControlStatus": "/rest/motorControlStatus",
    "inputMapperSettings": "/rest/inputMapperSettings",
    "rainSensorStatus": "/rest/rainSensorStatus",
    "rainSensorSettings": "/rest/rainSensorSettings",
    "twilightSwitchStatus": "/rest/twilightSwitchStatus",
    "twilightSwitchSettings": "/rest/twilightSwitchSettings"
}

last_signal_state = {}

# ================== HELPERS ==================

def run_cmd(cmd):
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        pass

def local_tz_label():
    """Return PST/PDT label based on local system time."""
    is_dst = time.localtime().tm_isdst
    try:
        return time.tzname[1] if is_dst else time.tzname[0]
    except Exception:
        return "LOCAL"

def UT_and_local():
    now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    utc = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"{utc} ({local_tz_label()})"

def load_sites():
    with open(CONFIG_FILE) as f:
        return json.load(f)["sites"]

def http_get_json(url):
    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def http_get_text(url):
    try:
        r = requests.get(url, timeout=8, headers={"accept": "text/plain"})
        r.raise_for_status()
        return r.text
    except Exception:
        return None

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def detect_changes(site, yyyymmdd, signals):
    global last_signal_state
    prev = last_signal_state.get(site)
    if not prev:
        last_signal_state[site] = signals
        return
    for field in ("roofStatus", "roofPosition", "operationMode"):
        if prev.get(field) != signals.get(field):
            log_event(site, yyyymmdd, f"{field}: {prev.get(field)} -> {signals.get(field)}")
    last_signal_state[site] = signals

def log_event(site, yyyymmdd, msg):
    log_dir = os.path.join(SAVE_ROOT, yyyymmdd, site, "dome")
    ensure_dir(log_dir)
    fn = os.path.join(log_dir, "roof_events.log")
    with open(fn, "a") as f:
        f.write(f"{UT_and_local()}  {site}: {msg}\n")
    print(f"[EVENT] {site}: {msg}")

def append_status(site, yyyymmdd, snapshot):
    log_dir = os.path.join(SAVE_ROOT, yyyymmdd, site, "dome")
    ensure_dir(log_dir)
    fn = os.path.join(log_dir, "roof_status.log")
    with open(fn, "a") as f:
        f.write(json.dumps(snapshot) + "\n")

def save_dome_logfile(site, yyyymmdd, text):
    if not text:
        return
    date_fmt = f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"
    log_dir = os.path.join(SAVE_ROOT, yyyymmdd, site, "dome")
    ensure_dir(log_dir)
    fn = os.path.join(log_dir, f"rest_logFile_{date_fmt}.log")
    with open(fn, "w") as f:
        f.write(text)
    print(f"[LOGFILE] Saved dome log for {site}")

def copy_to_cylon(name, bundle):
    tmp = f"/tmp/{name}.json"
    ensure_dir("/tmp")
    with open(tmp, "w") as f:
        json.dump(bundle, f, indent=2)

    REMOTE_DIR  = "/web/panoseti-palomar/current"
    REMOTE_DIR2 = "web/panoseti-palomar/current"

    # Upload JSON file (no mkdir here ? avoid permission error)
    run_cmd(f"scp -l {BANDWIDTH_LIMIT} {tmp} {REMOTE_SERVER}:{REMOTE_DIR}/")

    # Fix permissions for web access
    run_cmd(f'ssh {REMOTE_SERVER} "chmod 644 {REMOTE_DIR2}/{name}.json || true"')

    print(f"[UPLOAD] {name}.json ? {REMOTE_DIR}")



# ================== GATTINI PARSER (IMPROVED) ==================

def get_gattini_dome_status():
    url = "https://sites.astro.caltech.edu/~mhankins/gattini/data/gattini_status_nf.txt"
    try:
        r = requests.get(url, timeout=7)
        r.raise_for_status()
        lines = r.text.upper().splitlines()
    except Exception:
        return {
            "roofSignalsStatus": {"roofPosition": "UNKNOWN", "roofStatus": "UNKNOWN", "operationMode": "UNKNOWN"},
            "gattini_state_message": "NO DATA"
        }

    roof_pos = "UNKNOWN"
    for L in lines:
        if "SHUTTERS CLOSED" in L:
            roof_pos = "CLOSED"
            break
        if "SHUTTERS OPEN" in L:
            roof_pos = "OPENED"
            break

    op_mode = "UNKNOWN"
    for L in lines:
        if "SCHEDULER ENABLED" in L and "YES" in L:
            op_mode = "NORMAL"
            break

    state_msg = next((line for line in r.text.splitlines() if "state" in line.lower()), "NO STATE")

    return {
        "roofSignalsStatus": {
            "roofPosition": roof_pos,
            "roofStatus": "STOPPED" if roof_pos in ("OPENED", "CLOSED") else "UNKNOWN",
            "operationMode": op_mode
        },
        "gattini_state_message": state_msg
    }

# ================== MAIN LOOP ==================

def main():
    sites = load_sites()
    print(f"[INFO] Monitoring {len(sites)} domes every {INTERVAL_SECONDS}s.")

    while True:
        now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        yyyymmdd = now.strftime("%Y%m%d")
        timestamp = now.isoformat(timespec="seconds") + "Z"

        bundle_status = {"timestamp": timestamp, "sites": {}}
        bundle_logs = {"timestamp": timestamp, "sites": {}}

        for site in sites:
            name = site["name"]
            host = site["host"]
            snapshot = {"timestamp": timestamp, "site": name}

            if name == "Gattini":
                snapshot.update(get_gattini_dome_status())
                bundle_status["sites"][name] = snapshot
                bundle_logs["sites"][name] = snapshot["gattini_state_message"]
                append_status(name, yyyymmdd, snapshot)
                print("[OK] Gattini handled")
                continue

            for key, endpoint in ENDPOINTS.items():
                snapshot[key] = http_get_json(f"http://{host}:{PORT}{endpoint}")

            date_fmt = f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"
            log_text = http_get_text(f"http://{host}:{PORT}/rest/logFile?day={date_fmt}")
            save_dome_logfile(name, yyyymmdd, log_text)
            snapshot["logFile"] = "SAVED"
            bundle_logs["sites"][name] = log_text or "NO DATA"

            detect_changes(name, yyyymmdd, snapshot.get("roofSignalsStatus") or {})
            append_status(name, yyyymmdd, snapshot)

            bundle_status["sites"][name] = snapshot
            print(f"[OK] Logged dome status for {name}")

        copy_to_cylon("dome_current", bundle_status)
        copy_to_cylon("dome_logs", bundle_logs)

        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.\n")
