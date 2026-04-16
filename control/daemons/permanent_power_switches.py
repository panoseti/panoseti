#!/usr/bin/env python3
import datetime
import json
import os
import time

import redis
import requests
import urllib3

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "capture_power", "power_switches.conf")
SAVE_BASE   = "/mnt/data11/data/palomar/L0"
CURRENT_DIR_REMOTE = "panoseti@132.239.146.24:/web/panoseti-palomar/current/"
CURRENT_DIR_LOCAL  = "/tmp/power_current.json"

POLL_INTERVAL = 5
urllib3.disable_warnings()

from typing import Any


def load_config() -> dict[str, dict[str, str]]:
    sites: dict[str, dict[str, str]] = {}
    with open(CONFIG_FILE) as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): 
                continue
            name, ip, user, pwd = [x.strip() for x in line.split(",")]
            sites[name] = dict(ip=ip, user=user, pwd=pwd)
    return sites


def fetch_outlets(site: dict[str, str]) -> list[dict[str, Any]]:
    """Return list of outlet dicts in normalized format."""
    url = f"http://{site['ip']}/restapi/relay/outlets/"
    r = requests.get(url, auth=(site['user'], site['pwd']), timeout=4, verify=False)
    print(f"user: {site['user']}")
    print(f"pwd: {site['pwd']}")
    r.raise_for_status()
    data = r.json()

    # Normalize:
    if isinstance(data, dict) and "outlets" in data:
        return list(data["outlets"])
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected outlet format from {site['ip']}: {type(data)}")


def extract_state(outlet: dict[str, Any]) -> bool:
    """Return True/False from any available boolean field."""
    return bool(
        outlet.get("physical_state") or
        outlet.get("state") or
        outlet.get("transient_state") or
        False
    )


def write_daily_log(site: str, outlets: list[dict[str, Any]]) -> None:
    """Append log to /mnt/data11/data/palomar/L0/YYYYMMDD/site/power/"""
    date = datetime.datetime.now(datetime.UTC).replace(tzinfo=None).strftime("%Y%m%d")
    dpath = os.path.join(SAVE_BASE, date, site, "power")
    os.makedirs(dpath, exist_ok=True)
    logpath = os.path.join(dpath, f"{site}_power_{date}.log")

    ts = datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat()
    with open(logpath, "a") as f:
        for o in outlets:
            name = o.get("name", "Outlet").replace(" ", "_")
            f.write(f"{ts} | {site} | {name} | {'ON' if extract_state(o) else 'OFF'}\n")


def write_current_json(all_sites_data: dict[str, Any]) -> None:
    """Write combined JSON file and upload to cylon"""
    all_sites_data["timestamp"] = datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat()

    with open(CURRENT_DIR_LOCAL, "w") as f:
        json.dump(all_sites_data, f, indent=2)

    os.system(f"scp -q {CURRENT_DIR_LOCAL} {CURRENT_DIR_REMOTE}")


def push_redis(r: redis.Redis, site: str, outlets: list[dict[str, Any]]) -> None:
    
    for o in outlets:
        ts = time.time()
        name = o.get("name", "Outlet").replace(" ", "_")
        # key = f"POWER|{site.upper()}|{name.upper()}"
        #key = f"POWER_{site.upper()}_{name.upper()}"
        
        #key = f"POWER"
        
        #r.hset(key, mapping={
        #    "Computer_UTC": ts,                     # numeric timestamp
        #    "state": 1 if extract_state(o) else 0,  # numeric ON/OFF for Influx
        #    "site": site.upper(),                   # TAG for queries
        #    "device": name.upper()                  # TAG for queries
        #})

        #r.hset(key, mapping={
        #    "Computer_UTC": ts,
        #    "POWER": 1 if extract_state(o) else 0
        #})
        #r.hset(key, mapping={
        #    "Computer_UTC": ts,
        #    "POWER": 1 if extract_state(o) else 0,
        #    "site": site.upper(),
        #    "device": name.upper()
        #})
        #r.hset(key, mapping={
        #    "Computer_UTC": ts,
        #    "POWER": "ON" if extract_state(o) else "OFF"
        #})
        key = f"POWER_{site.upper()}_{name.upper()}"

        r.hset(key, mapping={
            "Computer_UTC": str(ts),
            "POWER": str(1 if extract_state(o) else 0),
            "site": site.upper(),
            "device": str(name).upper()
        })
        time.sleep(0.005)  # optional: tiny delay to ensure unique timestamps



def main() -> None:
    sites = load_config()
    print(f"[capture_power_switches] Using config: {CONFIG_FILE}")
    print(f"[capture_power_switches] Polling every {POLL_INTERVAL} seconds")

    r = redis.Redis()

    while True:
        all_sites_data = {}
        for site_name, site in sites.items():
            try:
                outlets = fetch_outlets(site)
                write_daily_log(site_name, outlets)
                push_redis(r, site_name, outlets)
                all_sites_data[site_name] = outlets
                print(f"[OK] {site_name}: {len(outlets)} outlets")
            except Exception as e:
                print(f"[ERROR] {site_name}: {e}")

        write_current_json(all_sites_data)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
