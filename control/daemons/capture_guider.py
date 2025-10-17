#!/usr/bin/env python3
import paramiko
import time
import os
from datetime import datetime
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# ---------------- CONFIG ----------------
RPI_HOST = "panoseti-gattini"                  # or IP address
RPI_USER = "stellarmate"
RPI_PASSWORD = "panoseti"
RPI_PORT = 5922
REMOTE_DIR = "/home/stellarmate/Pictures"  # Ekos capture directory
TRAIN_NAME = "Primary"                     # Optical train name in Ekos
WAIT_SECONDS = 6                           # Wait after Ekos capture
LOOP_INTERVAL_MIN = 1                      # Capture every N minutes (user-defined)
# ----------------------------------------

def qdbus(ssh, command):
    """Run a qdbus command remotely and return its stdout."""
    stdin, stdout, stderr = ssh.exec_command(command)
    return stdout.read().decode().strip()

def convert_fits_to_png(fits_path):
    """Convert a FITS file into PNG using astropy and matplotlib."""
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
        if data is None:
            print(f"?? No image data in {fits_path}")
            return None

        data = np.nan_to_num(data)
        vmin, vmax = np.percentile(data, (1, 99))
        data = np.clip(data, vmin, vmax)
        data = (data - vmin) / (vmax - vmin)

        png_path = fits_path.replace(".fits", ".png").replace(".fit", ".png")
        plt.imsave(png_path, data, cmap="gray", origin="lower")
        print(f"??  FITS converted to PNG ? {png_path}")
        return png_path
    except Exception as e:
        print(f"? FITS conversion failed: {e}")
        return None

def ekos_capture_once():
    """Trigger a single Ekos capture and download the FITS + PNG locally."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(RPI_HOST, port=RPI_PORT, username=RPI_USER, password=RPI_PASSWORD)

    print("\n? Starting new capture sequence")
    print("? Reading Ekos Capture properties ...")
    camera = qdbus(ssh, 'qdbus org.kde.kstars /KStars/Ekos/Capture '
                        'org.freedesktop.DBus.Properties.Get org.kde.kstars.Ekos.Capture camera')
    optical_train = qdbus(ssh, 'qdbus org.kde.kstars /KStars/Ekos/Capture '
                               'org.freedesktop.DBus.Properties.Get org.kde.kstars.Ekos.Capture opticalTrain')
    print(f"? Camera: {camera}")
    print(f"? Optical Train: {optical_train}")

    # --- Trigger capture ---
    print(f"? Capture via D-Bus (train = '{TRAIN_NAME}') ...")
    qdbus(ssh, f'qdbus org.kde.kstars /KStars/Ekos/Capture '
               f'org.kde.kstars.Ekos.Capture.start \"{TRAIN_NAME}\"')

    print(f"? Waiting {WAIT_SECONDS}s for Ekos to save image ...")
    time.sleep(WAIT_SECONDS)

    # --- Query filename ---
    preview_name = qdbus(ssh, 'qdbus org.kde.kstars /KStars/Ekos/Capture '
                              'org.kde.kstars.Ekos.Capture.getJobPreviewFileName')
    if not preview_name or "[NATIVE]" in preview_name:
        preview_name = preview_name.replace("[NATIVE]", "fits") if preview_name else ""
        if not preview_name:
            stdin, stdout, stderr = ssh.exec_command(
                f'find {REMOTE_DIR} -type f -name "*.fit*" -printf "%T@ %p\\n" | sort -nr | head -n 1 | cut -d" " -f2'
            )
            preview_name = stdout.read().decode().strip()

    if not preview_name:
        print("??  No FITS file found. Check Ekos capture directory.")
        ssh.close()
        return

    # --- Build local directory ---
    date_str = datetime.utcnow().strftime("%Y%m%d")
    local_dir = f"/mnt/data11/data/palomar/L0/{date_str}/guider"
    os.makedirs(local_dir, exist_ok=True)

    # --- Filename format ---
    host_tag = RPI_HOST.split(".")[0]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    local_fits = os.path.join(local_dir, f"guider_{host_tag}_{timestamp}.fits")

    # --- Download ---
    print(f"??  Downloading {preview_name} ? {local_fits}")
    sftp = ssh.open_sftp()
    try:
        sftp.get(preview_name, local_fits)
        print(f"? FITS saved as {local_fits}")
        convert_fits_to_png(local_fits)
    except FileNotFoundError:
        print(f"?? Remote file not found: {preview_name}")
    finally:
        sftp.close()
        ssh.close()

def main():
    """Main loop: capture every LOOP_INTERVAL_MIN minutes."""
    print(f"? Starting continuous guider capture loop (every {LOOP_INTERVAL_MIN} min)")
    while True:
        start_time = time.time()
        ekos_capture_once()
        elapsed = time.time() - start_time
        sleep_time = max(0, LOOP_INTERVAL_MIN * 60 - elapsed)
        print(f"? Sleeping {sleep_time/60:.1f} min before next capture ...")
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
