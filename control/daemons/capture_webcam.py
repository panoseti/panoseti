#!/usr/bin/env python3
import requests
from requests.auth import HTTPDigestAuth
from datetime import datetime, timezone
import os
import time
import socket

# ---------------- CONFIGURATION ----------------
CAMERA_IP = "panoseti-gattini:8080"      # Change to your camera IP or hostname
USERNAME = "admin"                       # Camera username
PASSWORD = "123456"               # Camera password
INTERVAL_SEC = 20                        # Capture interval in seconds
BASE_DIR = "/mnt/data11/data/palomar/L0" # Base directory
# ------------------------------------------------

# Camera snapshot endpoint
URL = f"http://{CAMERA_IP}/cgi-bin/snapshot.cgi"

# Hostname for filename tag
RPI_HOST = socket.gethostname()

def make_save_dir():
    """Return the directory path based on current UTC date."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    save_dir = os.path.join(BASE_DIR, date_str, "webcam")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def grab_snapshot():
    """Fetch and save one snapshot from the IP camera."""
    save_dir = make_save_dir()
    try:
        response = requests.get(URL, auth=HTTPDigestAuth(USERNAME, PASSWORD), timeout=10)
        response.raise_for_status()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"webcam_{RPI_HOST}_{timestamp}.jpg")

        with open(filename, "wb") as f:
            f.write(response.content)

        print(f"[{timestamp} UTC] ? Saved snapshot: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"[!] ? Error fetching snapshot: {e}")

def main():
    print(f"Starting webcam capture every {INTERVAL_SEC} sec from {CAMERA_IP} ...")
    while True:
        grab_snapshot()
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
