#!/usr/bin/env python3
import os
import time
import base64
import json
import argparse
import re
import shutil
import subprocess
import numpy as np
from datetime import datetime
from astropy.io import fits
import matplotlib.pyplot as plt
import paramiko

# ---------------- CONFIG ----------------
CONFIG_FILE = "/home/obs/panoseti_mount/panoseti/control/daemons/capture_guider/sites.conf"
REMOTE_DIR = "/home/panoseti/Pictures"
TRAIN_NAME = "Primary"
WAIT_SECONDS = 15
LOOP_INTERVAL_MIN = 2
LOCAL_BASE = "/mnt/data11/data/palomar/L0"
OPTIPNG_LEVEL = 5  # 0..7 (lossless). 5 is a good speed/ratio balance.
# ----------------------------------------


def load_sites(config_file):
    """Read sites.conf and return list of dicts: name, host, user, password, port."""
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


def ssh_connect(site):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        site["host"],
        port=site["port"],
        username=site["user"],
        password=site["password"],
        timeout=5
    )
    return ssh


def ssh_exec(ssh, cmd, timeout=None):
    """Execute remote command, return (stdout, stderr) strings."""
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace").strip()
    err = stderr.read().decode(errors="replace").strip()
    return out, err


# ==============================
# PNG lossless recompression
# ==============================
def optipng_available():
    """Return True if optipng is available in PATH on THIS machine (local)."""
    return shutil.which("optipng") is not None


def recompress_png_lossless(png_path, level=OPTIPNG_LEVEL):
    """Losslessly recompress a PNG in-place using optipng."""
    if not optipng_available():
        print(f"?? optipng not found, skipping PNG recompression: {png_path}")
        return

    try:
        size_before = os.path.getsize(png_path)
        subprocess.run(
            ["optipng", f"-o{level}", png_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
        size_after = os.path.getsize(png_path)
        if size_after < size_before:
            print(
                f"? PNG recompressed (lossless): {png_path} "
                f"({size_before/1024:.1f} KB -> {size_after/1024:.1f} KB)"
            )
        else:
            print(f"? PNG recompressed (lossless): {png_path} (no gain)")
    except Exception as e:
        print(f"?? optipng failed for {png_path}: {e}")


def convert_fits_to_png(fits_path):
    """Convert FITS to PNG and then losslessly recompress PNG (optipng)."""
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data

        if data is None:
            print(f"?? No image data in {fits_path}")
            return

        data = np.nan_to_num(data)
        vmin, vmax = np.percentile(data, (1, 99))
        if vmax <= vmin:
            print(f"?? Invalid percentile range for {fits_path} (vmin={vmin}, vmax={vmax})")
            return

        data = np.clip(data, vmin, vmax)
        data = (data - vmin) / (vmax - vmin)

        png_path = fits_path.replace(".fits", ".png").replace(".fit", ".png")
        plt.imsave(png_path, data, cmap="gray", origin="lower")
        print(f"? FITS converted to PNG -> {png_path}")

        # --- NEW: lossless PNG recompression ---
        recompress_png_lossless(png_path, level=OPTIPNG_LEVEL)

    except Exception as e:
        print(f"? FITS conversion failed: {e}")


# ==============================
# Ekos Guide status / control
# ==============================
def ekos_get_guide_status(ssh):
    """
    Return (status_int_or_None, raw_stdout, raw_stderr).

    NOTE: status integer mapping depends on KStars/Ekos version.
    We only display the raw integer for traceability.
    """
    cmd = (
        "qdbus org.kde.kstars /KStars/Ekos/Guide "
        "org.freedesktop.DBus.Properties.Get org.kde.kstars.Ekos.Guide status"
    )
    out, err = ssh_exec(ssh, cmd)

    status = None
    try:
        nums = re.findall(r"-?\d+", out)
        if nums:
            status = int(nums[-1])
    except Exception:
        status = None

    return status, out, err


def ekos_guide_suspend(ssh):
    cmd = "qdbus org.kde.kstars /KStars/Ekos/Guide org.kde.kstars.Ekos.Guide.suspend"
    return ssh_exec(ssh, cmd)


def ekos_guide_resume(ssh):
    cmd = "qdbus org.kde.kstars /KStars/Ekos/Guide org.kde.kstars.Ekos.Guide.resume"
    return ssh_exec(ssh, cmd)


# ==============================
# PHD2 helpers (optional)
# ==============================
def phd2_running_via_ssh(site):
    try:
        ssh = ssh_connect(site)
        cmd = "echo '{\"method\":\"get_app_state\",\"id\":1,\"jsonrpc\":\"2.0\"}' | nc -w 2 localhost 4400"
        out, err = ssh_exec(ssh, cmd)
        ssh.close()
        return ("PHDVersion" in out) or ('"result"' in out)
    except Exception:
        return False


def capture_phd2_once(site):
    print(f"? [{site['name']}] Capturing via PHD2 ...")
    try:
        ssh = ssh_connect(site)
        cmd = "echo '{\"method\":\"get_star_image\",\"id\":1,\"jsonrpc\":\"2.0\"}' | nc -w 5 localhost 4400"
        output, err = ssh_exec(ssh, cmd)
        ssh.close()

        reply = None
        for line in output.splitlines():
            try:
                reply = json.loads(line)
            except Exception:
                continue

        if not reply or "result" not in reply or "image" not in reply["result"]:
            print(f"?? [{site['name']}] No image data from PHD2.")
            if err:
                print(f"?? [{site['name']}] PHD2 stderr: {err}")
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


# ==============================
# Ekos capture (default)
# ==============================
def capture_ekos_once(site):
    print(f"? [{site['name']}] Capturing via Ekos (suspend -> capture -> resume) ...")
    ssh = None

    try:
        ssh = ssh_connect(site)

        # Status before
        st, raw, err = ekos_get_guide_status(ssh)
        if st is None:
            print(f"?? [{site['name']}] Guide status read failed. Raw: '{raw}' Err: '{err}'")
        else:
            print(f"? [{site['name']}] Guide status BEFORE suspend: {st}")

        # Suspend guiding
        print(f"? [{site['name']}] Suspending guiding...")
        out, serr = ekos_guide_suspend(ssh)
        if out:
            print(f"? [{site['name']}] suspend() returned: {out}")
        if serr:
            print(f"?? [{site['name']}] suspend() stderr: {serr}")

        # Status after suspend
        st2, raw2, err2 = ekos_get_guide_status(ssh)
        if st2 is None:
            print(f"?? [{site['name']}] Guide status AFTER suspend read failed. Raw: '{raw2}' Err: '{err2}'")
        else:
            print(f"? [{site['name']}] Guide status AFTER suspend: {st2}")

        # Start capture
        print(f"? [{site['name']}] Starting Ekos Capture for optical train '{TRAIN_NAME}'...")
        out_cap, err_cap = ssh_exec(
            ssh,
            f'qdbus org.kde.kstars /KStars/Ekos/Capture '
            f'org.kde.kstars.Ekos.Capture.start "{TRAIN_NAME}"'
        )
        if out_cap:
            print(f"? [{site['name']}] Capture.start returned: {out_cap}")
        if err_cap:
            print(f"?? [{site['name']}] Capture.start stderr: {err_cap}")

        # Wait for the file
        print(f"? [{site['name']}] Waiting {WAIT_SECONDS}s for FITS to appear in {REMOTE_DIR} ...")
        time.sleep(WAIT_SECONDS)

        # Find newest FITS
        find_cmd = (
            f'find {REMOTE_DIR} -type f -name "*.fit*" '
            '-printf "%T@ %p\\n" | sort -nr | head -n 1 | cut -d" " -f2'
        )
        preview_name, find_err = ssh_exec(ssh, find_cmd)
        if not preview_name:
            print(f"?? [{site['name']}] No FITS found in {REMOTE_DIR}.")
            if find_err:
                print(f"?? [{site['name']}] find stderr: {find_err}")
            return

        # Download
        date_str = datetime.utcnow().strftime("%Y%m%d")
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        local_dir = os.path.join(LOCAL_BASE, date_str, site["name"], "guider")
        os.makedirs(local_dir, exist_ok=True)

        local_fits = os.path.join(local_dir, f"guider_{site['name']}_{ts}.fits")
        print(f"? [{site['name']}] Downloading: {preview_name} -> {local_fits}")

        sftp = ssh.open_sftp()
        sftp.get(preview_name, local_fits)
        sftp.close()

        print(f"? [{site['name']}] Ekos FITS saved: {local_fits}")
        convert_fits_to_png(local_fits)

    except Exception as e:
        print(f"? [{site['name']}] Ekos capture failed: {e}")

    finally:
        if ssh is not None:
            # Resume guiding (best-effort)
            try:
                st3, raw3, err3 = ekos_get_guide_status(ssh)
                if st3 is None:
                    print(f"?? [{site['name']}] Guide status BEFORE resume read failed. Raw: '{raw3}' Err: '{err3}'")
                else:
                    print(f"? [{site['name']}] Guide status BEFORE resume: {st3}")

                print(f"? [{site['name']}] Resuming guiding...")
                out, serr = ekos_guide_resume(ssh)
                if out:
                    print(f"? [{site['name']}] resume() returned: {out}")
                if serr:
                    print(f"?? [{site['name']}] resume() stderr: {serr}")

                st4, raw4, err4 = ekos_get_guide_status(ssh)
                if st4 is None:
                    print(f"?? [{site['name']}] Guide status AFTER resume read failed. Raw: '{raw4}' Err: '{err4}'")
                else:
                    print(f"? [{site['name']}] Guide status AFTER resume: {st4}")

            except Exception as e:
                print(f"?? [{site['name']}] Guide resume/status check failed: {e}")

            try:
                ssh.close()
            except Exception:
                pass


# ==============================
# Main loop / CLI
# ==============================
def parse_args():
    p = argparse.ArgumentParser(
        description="Capture guider images periodically from remote sites. Ekos by default; optional PHD2."
    )
    p.add_argument(
        "--use-phd2",
        action="store_true",
        help="Use PHD2 get_star_image if PHD2 is running on the remote host; otherwise fall back to Ekos."
    )
    return p.parse_args()


def main():
    args = parse_args()
    sites = load_sites(CONFIG_FILE)

    print(f"? Active sites: {[s['name'] for s in sites]}")
    print(f"? Mode: {'PHD2 (if running) else Ekos' if args.use_phd2 else 'Ekos only (default)'}")
    print(f"? Local optipng: {'FOUND' if optipng_available() else 'NOT FOUND'} (lossless PNG recompression)")

    while True:
        start = time.time()

        for site in sites:
            if args.use_phd2 and phd2_running_via_ssh(site):
                capture_phd2_once(site)
            else:
                capture_ekos_once(site)

        elapsed = time.time() - start
        sleep_time = max(0, LOOP_INTERVAL_MIN * 60 - elapsed)
        print(f"\n? Sleeping {sleep_time/60:.1f} min...\n")
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()

