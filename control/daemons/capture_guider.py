#!/usr/bin/env python3
import os
import time
import base64
import numpy as np
from datetime import datetime
from astropy.io import fits
import matplotlib.pyplot as plt
import paramiko

# ---------------- CONFIG ----------------
CONFIG_FILE = "/home/obs/panoseti_mount/panoseti/control/daemons/capture_guider/sites.conf"
REMOTE_DIR = "/home/stellarmate/Pictures"
TRAIN_NAME = "Primary"
WAIT_SECONDS = 10
LOOP_INTERVAL_MIN = 1
LOCAL_BASE = "/mnt/data11/data/palomar/L0"
# ----------------------------------------

def load_sites(config_file):
    """Read sites.conf and return list of dicts."""
    sites = []
    with open(config_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            name, host, user, password, port = parts
            sites.append({
                "name": name,
                "host": host,
                "user": user,
                "password": password,
                "port": int(port)
            })
    return sites

def convert_fits_to_png(fits_path):
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
        if data is None:
            print(f"?? No image data in {fits_path}")
            return
        data = np.nan_to_num(data)
        vmin, vmax = np.percentile(data, (1, 99))
        data = np.clip(data, vmin, vmax)
        data = (data - vmin) / (vmax - vmin)
        png_path = fits_path.replace(".fits", ".png").replace(".fit", ".png")
        plt.imsave(png_path, data, cmap="gray", origin="lower")
        print(f"? FITS converted to PNG ? {png_path}")
    except Exception as e:
        print(f"? FITS conversion failed: {e}")

def ssh_connect(site):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(site["host"], port=site["port"], username=site["user"],
                password=site["password"], timeout=5)
    return ssh

def phd2_running_via_ssh(site):
    try:
        ssh = ssh_connect(site)
        cmd = "echo '{\"method\":\"get_app_state\",\"id\":1,\"jsonrpc\":\"2.0\"}' | nc -w 2 localhost 4400"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        output = stdout.read().decode().strip()
        ssh.close()
        return "PHDVersion" in output or '"result"' in output
    except Exception:
        return False

def capture_phd2_once(site):
    print(f"? [{site['name']}] Capturing via PHD2 ...")
    try:
        ssh = ssh_connect(site)
        cmd = "echo '{\"method\":\"get_star_image\",\"id\":1,\"jsonrpc\":\"2.0\"}' | nc -w 5 localhost 4400"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        output = stdout.read().decode()
        ssh.close()
        reply = None
        for line in output.splitlines():
            try:
                reply = json.loads(line)
            except Exception:
                continue
        if not reply or "result" not in reply or "image" not in reply["result"]:
            print(f"?? [{site['name']}] No image data from PHD2.")
            return
        img_b64 = reply["result"]["image"]
        w = reply["result"]["width"]
        h = reply["result"]["height"]
        arr = np.frombuffer(base64.b64decode(img_b64), dtype=np.uint16).reshape(h, w)

        date_str = datetime.utcnow().strftime("%Y%m%d")
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        local_dir = os.path.join(LOCAL_BASE, date_str, site["name"], "guider")
        os.makedirs(local_dir, exist_ok=True)
        fits_path = os.path.join(local_dir, f"guider_{site['name']}_{ts}.fits")
        fits.writeto(fits_path, arr, overwrite=True)
        print(f"? [{site['name']}] FITS saved: {fits_path}")
        convert_fits_to_png(fits_path)
    except Exception as e:
        print(f"? [{site['name']}] PHD2 capture failed: {e}")

def capture_ekos_once(site):
    print(f"? [{site['name']}] Capturing via Ekos ...")
    try:
        ssh = ssh_connect(site)
        ssh.exec_command(f'qdbus org.kde.kstars /KStars/Ekos/Capture '
                         f'org.kde.kstars.Ekos.Capture.start \"{TRAIN_NAME}\"')
        time.sleep(WAIT_SECONDS)
        stdin, stdout, stderr = ssh.exec_command(
            f'find {REMOTE_DIR} -type f -name "*.fit*" '
            '-printf "%T@ %p\\n" | sort -nr | head -n 1 | cut -d" " -f2')
        preview_name = stdout.read().decode().strip()
        if not preview_name:
            print(f"?? [{site['name']}] No FITS found.")
            ssh.close()
            return
        date_str = datetime.utcnow().strftime("%Y%m%d")
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        local_dir = os.path.join(LOCAL_BASE, date_str, site["name"], "guider")
        os.makedirs(local_dir, exist_ok=True)
        local_fits = os.path.join(local_dir, f"guider_{site['name']}_{ts}.fits")
        sftp = ssh.open_sftp()
        sftp.get(preview_name, local_fits)
        sftp.close()
        ssh.close()
        print(f"? [{site['name']}] Ekos FITS saved: {local_fits}")
        convert_fits_to_png(local_fits)
    except Exception as e:
        print(f"? [{site['name']}] Ekos capture failed: {e}")

def main():
    sites = load_sites(CONFIG_FILE)
    print(f"? Active sites: {[s['name'] for s in sites]}")
    while True:
        start = time.time()
        for site in sites:
            if phd2_running_via_ssh(site):
                capture_phd2_once(site)
            else:
                capture_ekos_once(site)
        elapsed = time.time() - start
        sleep_time = max(0, LOOP_INTERVAL_MIN * 60 - elapsed)
        print(f"\n? Sleeping {sleep_time/60:.1f} min...\n")
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
