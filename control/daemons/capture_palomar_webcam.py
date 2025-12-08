#!/usr/bin/env python3
import requests
import time
import os
from datetime import datetime, timezone
import subprocess

# ---------------- CONFIG ----------------
IMAGE_URL = "https://algol.palomar.caltech.edu/instruments/allsky/AllSkyCurrentImage.JPG"
BASE_DIR = "/mnt/data11/data/palomar/L0"   # Base directory
INTERVAL = 30                              # Seconds between saves (user-defined)

# Cylon upload config
REMOTE_SERVER = "panoseti@132.239.146.24"
REMOTE_DIR = "/web/panoseti-palomar/current"
REMOTE_DIR2 = "/web/panoseti-palomar/current"    # Used only for chmod
BANDWIDTH_LIMIT = 40000        # kbit/s scp limit
# ----------------------------------------

def run_cmd(cmd):
    """Run a shell command and show if error occurs."""
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"[!] Command failed: {cmd}")

def make_save_dir():
    """Return today's (UTC) save directory and create it if missing."""
    utc_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    save_dir = os.path.join(BASE_DIR, utc_date, "allsky")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def download_image(save_dir):
    """Download the All-Sky image and save it with a UTC timestamp."""
    try:
        r = requests.get(IMAGE_URL, timeout=10)
        r.raise_for_status()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"allsky_{timestamp}.jpg")

        with open(filename, "wb") as f:
            f.write(r.content)

        print(f"[+] Saved: {filename}")

        # Also send to cylon
        upload_latest_to_cylon(filename)

    except Exception as e:
        print(f"[!] Error downloading image: {e}")

def upload_latest_to_cylon(local_path):
    """Copy latest All-Sky image to cylon as allsky_current.jpg."""
    tmp = "/tmp/allsky_current.jpg"
    try:
        # Copy file locally first
        run_cmd(f"cp '{local_path}' '{tmp}'")

        # Upload to cylon (no mkdir needed)
        run_cmd(f"scp -l {BANDWIDTH_LIMIT} '{tmp}' {REMOTE_SERVER}:{REMOTE_DIR}/allsky_current.jpg")

        # Fix permissions
        run_cmd(f'ssh {REMOTE_SERVER} "chmod 644 {REMOTE_DIR2}/allsky_current.jpg || true"')

        print("? Updated allsky_current.jpg on cylon")

    except Exception as e:
        print(f"[!] Failed to upload to cylon: {e}")

def main():
    print(f"Fetching All-Sky images every {INTERVAL} seconds (UTC timestamps).")
    print("Press Ctrl-C to stop.")
    while True:
        save_dir = make_save_dir()
        download_image(save_dir)
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
