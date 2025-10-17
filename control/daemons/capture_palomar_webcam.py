#!/usr/bin/env python3
import requests
import time
import os
from datetime import datetime, timezone

# ---------------- CONFIG ----------------
IMAGE_URL = "https://algol.palomar.caltech.edu/instruments/allsky/AllSkyCurrentImage.JPG"
BASE_DIR = "/mnt/data11/data/palomar/L0"   # Base directory
INTERVAL = 30                              # Seconds between saves (user-defined)
# ----------------------------------------

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
    except Exception as e:
        print(f"[!] Error downloading image: {e}")

def main():
    print(f"Fetching All-Sky images every {INTERVAL} seconds (UTC timestamps).")
    print("Press Ctrl-C to stop.")
    while True:
        save_dir = make_save_dir()
        download_image(save_dir)
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
