#!/usr/bin/env python3
import requests
from requests.auth import HTTPDigestAuth
from datetime import datetime, timezone
import os
import time
import socket
import subprocess

# ---------------- CONFIGURATION ----------------
CONFIG_FILE = "/home/obs/panoseti_mount/panoseti/control/daemons/capture_webcam/sites_webcam.conf"
USERNAME = "admin"
PASSWORD = "123456"
INTERVAL_SEC = 20

BASE_DIR_ROOT = "/mnt/data11/data/palomar/L0"

# Remote sync config (unchanged, working!)
BANDWIDTH_LIMIT = 40000
REMOTE_SERVER = "panoseti@132.239.146.24"
REMOTE_WEBCAM_DIR = "/web/panoseti-palomar/current"
REMOTE_WEBCAM_DIR2 = "/web/panoseti-palomar/current"

RPI_HOST = socket.gethostname()

# ---------------- HELPERS ----------------
def run_command(cmd):
    print(f"[CMD] {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("  ? Done.")
    except subprocess.CalledProcessError:
        print("  ? Failed.")

def load_sites(config_path):
    sites = []
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, ip = [x.strip() for x in line.split(",")]
            sites.append({"name": name, "ip": ip})
    return sites

def make_save_dir(site_name):
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    save_dir = os.path.join(BASE_DIR_ROOT, date_str, site_name, "webcam")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

# ---------------- CAPTURE + UPLOAD ----------------
def capture_and_upload(site):
    site_name = site["name"]
    camera_ip = site["ip"]

    url = f"http://{camera_ip}/cgi-bin/snapshot.cgi"
    save_dir = make_save_dir(site_name)

    try:
        r = requests.get(url, auth=HTTPDigestAuth(USERNAME, PASSWORD), timeout=10)
        r.raise_for_status()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        local_filename = os.path.join(save_dir, f"webcam_{site_name}_{RPI_HOST}_{timestamp}.jpg")

        with open(local_filename, "wb") as f:
            f.write(r.content)

        print(f"[{timestamp} UTC] ? {site_name}: Saved {local_filename}")

        # Also update the web "current image" version (this is the key change!)
        current_name = f"{site_name}-webcam-current.jpg"
        with open(current_name, "wb") as f:
            f.write(r.content)

        # Upload to Synology (no mkdir!)
        run_command(f"scp -l {BANDWIDTH_LIMIT} {current_name} {REMOTE_SERVER}:{REMOTE_WEBCAM_DIR}/")

        # Fix permissions for web
        run_command(f'ssh {REMOTE_SERVER} "chmod 644 {REMOTE_WEBCAM_DIR2}/{current_name} || true"')

    except Exception as e:
        print(f"[!] ?? {site_name}: Capture/upload failed: {e}")

# ---------------- MAIN LOOP ----------------
def main():
    sites = load_sites(CONFIG_FILE)
    print(f"Starting webcam capture + web sync every {INTERVAL_SEC}s for {len(sites)} sites...")
    for s in sites:
        print(f"  ? {s['name']} @ {s['ip']}")

    while True:
        for site in sites:
            capture_and_upload(site)
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
