#!/usr/bin/env python3
import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import paramiko
from astropy.io import fits

# ---------------- CONFIG ----------------
CONFIG_FILE = "/home/obs/panoseti_mount/panoseti/control/daemons/capture_guider/sites.conf"
REMOTE_DIR = "/home/panoseti/Pictures"
TRAIN_NAME = "Primary"
WAIT_SECONDS = 15
LOOP_INTERVAL_MIN = 3
LOCAL_BASE = "/mnt/data11/data/palomar/L0"
OPTIPNG_LEVEL = 5  # 0..7 (lossless). 5 is a good speed/ratio balance.

# --- NEW: optional periodic Align (can be changed easily) ---
ALIGN_EVERY_N_ITERATIONS = 4          # do Align every N outer-loop iterations
ALIGN_HA_THRESHOLD_HOURS = 0.5        # 0h30m = 0.5 hours
ALIGN_MAX_WAIT_SECONDS_FACTOR = 3     # keep same WAIT_SECONDS*3 style for align waits if needed
# NOTE: Ekos "Slew to Target" action enum can vary by version/build.
# If your Ekos profile already has solver action set to "Slew to Target", captureAndSolve() will honor it.
# If you want to force it, set this to an int that matches your build, otherwise leave FORCE_ALIGN_SOLVER_ACTION=None.
FORCE_ALIGN_SOLVER_ACTION = 2  # e.g., 2 (if your build uses 2 for "Slew to Target")
# ----------------------------------------


# Tracks last-downloaded remote file per site (extra safety against repeats)
LAST_EKOS_REMOTE: dict[str, Any] = {}
  # site_name -> remote_path


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
        timeout=10
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
# OPTIONAL: mount tracking + HA checks (best-effort, skip if unknown)
# ==============================
def _extract_last_number(s):
    nums = re.findall(r"-?\d+(?:\.\d+)?", s or "")
    if not nums:
        return None
    try:
        return float(nums[-1])
    except Exception:
        return None


def _parse_ha_to_hours(ha_str):
    """
    Parse HA into hours. Accepts:
      - float-like ("-0.62")
      - "HH:MM:SS" (with optional sign)
      - "HhMmSs" rough patterns
    Returns float hours or None.
    """
    if ha_str is None:
        return None
    s = ha_str.strip()
    if not s:
        return None

    # direct float?
    try:
        return float(s)
    except Exception:
        pass

    # match +/-HH:MM:SS
    m = re.search(r"([+-]?\d+)\s*:\s*(\d+)\s*:\s*(\d+(?:\.\d+)?)", s)
    if m:
        hh = float(m.group(1))
        mm = float(m.group(2))
        ss = float(m.group(3))
        sign = -1.0 if hh < 0 else 1.0
        hh_abs = abs(hh)
        return sign * (hh_abs + mm / 60.0 + ss / 3600.0)

    # match +/-HH:MM
    m = re.search(r"([+-]?\d+)\s*:\s*(\d+)", s)
    if m:
        hh = float(m.group(1))
        mm = float(m.group(2))
        sign = -1.0 if hh < 0 else 1.0
        hh_abs = abs(hh)
        return sign * (hh_abs + mm / 60.0)

    return None


def ekos_mount_is_tracking_and_ha_hours(ssh):
    """
    Best-effort retrieval of:
      - tracking state (bool or None)
      - HA in hours (float or None)
    If we cannot determine either, returns (None, None) and caller should skip alignment.
    """
    tracking = None
    ha_hours = None

    # Try a few common Ekos Mount DBus property names (varies by build).
    # If these are not present on your system, this will just fail gracefully.
    candidate_props = [
        ("org.kde.kstars.Ekos.Mount", "tracking"),
        ("org.kde.kstars.Ekos.Mount", "isTracking"),
        ("org.kde.kstars.Ekos.Mount", "status"),
        ("org.kde.kstars.Ekos.Mount", "hourAngle"),
        ("org.kde.kstars.Ekos.Mount", "ha"),
    ]

    # We only try; if nothing works, return None values.
    for iface, prop in candidate_props:
        cmd = f"qdbus org.kde.kstars /KStars/Ekos/Mount org.freedesktop.DBus.Properties.Get {iface} {prop}"
        out, err = ssh_exec(ssh, cmd)
        if err and "Unknown" in err:
            continue
        if not out:
            continue

        # Interpret tracking-ish properties
        if prop in ("tracking", "isTracking"):
            # could be "true"/"false" or QVariant wrappers
            lo = out.lower()
            if "true" in lo:
                tracking = True
            elif "false" in lo:
                tracking = False

        if prop in ("status",):
            # status int mapping unknown; we can't safely map -> do not infer tracking here
            pass

        # Interpret HA-ish properties
        if prop in ("hourAngle", "ha"):
            ha_hours = _parse_ha_to_hours(out)

    return tracking, ha_hours


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

        date_str = datetime.now(UTC).replace(tzinfo=None).strftime("%Y%m%d")
        ts = datetime.now(UTC).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
        local_dir = os.path.join(LOCAL_BASE, date_str, site["name"], "guider")
        os.makedirs(local_dir, exist_ok=True)

        fits_path = os.path.join(local_dir, f"guider_{site['name']}_{ts}.fits")
        fits.writeto(fits_path, arr, overwrite=True)

        print(f"? [{site['name']}] FITS saved: {fits_path}")
        convert_fits_to_png(fits_path)

    except Exception as e:
        print(f"? [{site['name']}] PHD2 capture failed: {e}")


# ==============================
# Robust Ekos FITS detection
# ==============================
def remote_find_newest_fits_with_mtime(ssh):
    """
    Return (mtime_float_or_None, path_or_empty, stderr).
    Uses recursive search under REMOTE_DIR.
    """
    cmd = (
        f'find {REMOTE_DIR} -type f -name "*.fit*" '
        '-printf "%T@ %p\\n" | sort -nr | head -n 1'
    )
    out, err = ssh_exec(ssh, cmd)
    if not out:
        return None, "", err

    # Expected: "<mtime> <path>"
    parts = out.split(" ", 1)
    if len(parts) != 2:
        return None, "", err

    try:
        mtime = float(parts[0])
    except Exception:
        mtime = None

    path = parts[1].strip()
    return mtime, path, err


def wait_for_new_fits(ssh, site_name, baseline_mtime, max_wait_seconds):
    """
    Poll until we see a FITS with mtime strictly greater than baseline_mtime,
    and not equal to the last downloaded path for this site (extra safety).
    Returns (new_mtime, new_path) or (None, "") on timeout.
    """
    poll_interval = 1.0
    deadline = time.time() + max_wait_seconds
    last_seen = LAST_EKOS_REMOTE.get(site_name)

    print(
        f"? [{site_name}] Waiting for new FITS (baseline mtime={baseline_mtime}) "
        f"max_wait={max_wait_seconds}s ..."
    )

    while time.time() < deadline:
        mtime, path, err = remote_find_newest_fits_with_mtime(ssh)
        if err:
            print(f"?? [{site_name}] find stderr: {err}")

        if path and (mtime is not None):
            if (baseline_mtime is None or mtime > baseline_mtime) and (path != last_seen):
                print(f"? [{site_name}] New FITS detected: mtime={mtime:.3f} path={path}")
                return mtime, path

        time.sleep(poll_interval)

    print(f"?? [{site_name}] Timed out waiting for a new FITS.")
    return None, ""


# ==============================
# OPTIONAL: Align (capture & solve with "Slew to Target")
# ==============================
# ==============================
# OPTIONAL: Align (capture & solve with "Slew to Target")  [PATCHED]
# ==============================
def _get_align_status_int(ssh):
    """
    Return (status_int_or_None, raw_stdout, raw_stderr) for Ekos Align.status
    """
    cmd = (
        "qdbus org.kde.kstars /KStars/Ekos/Align "
        "org.freedesktop.DBus.Properties.Get org.kde.kstars.Ekos.Align status"
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


def wait_for_align_done(ssh, site_name, max_wait_seconds):
    """
    Wait until Align is no longer busy.
    We treat status==0 as "idle/done". We don't rely on exact enum mappings.
    Returns True if done, False on timeout.
    """
    deadline = time.time() + max_wait_seconds
    poll_interval = 1.0

    # If we're already idle, nothing to wait for
    st0, raw0, err0 = _get_align_status_int(ssh)
    if st0 is not None:
        print(f"? [{site_name}] Align status BEFORE wait: {st0}")
        if st0 == 0:
            return True
    else:
        print(f"?? [{site_name}] Align status read failed BEFORE wait. Raw: '{raw0}' Err: '{err0}'")

    while time.time() < deadline:
        st, raw, err = _get_align_status_int(ssh)
        if st is None:
            print(f"?? [{site_name}] Align status read failed. Raw: '{raw}' Err: '{err}'")
            time.sleep(poll_interval)
            continue

        # Robust rule: 0 means idle/done in Ekos modules.
        if st == 0:
            print(f"? [{site_name}] Align finished (status={st})")
            return True

        print(f"? [{site_name}] Align running (status={st}) ...")
        time.sleep(poll_interval)

    print(f"?? [{site_name}] Align did not finish before timeout ({max_wait_seconds}s).")
    return False


def maybe_align_winter(ssh, site_name, iteration_idx, do_align):
    """
    Called immediately AFTER guiding is suspended.
    Performs Align (captureAndSolve) ONLY IF:
      - do_align option enabled
      - site_name == "Winter"
      - every ALIGN_EVERY_N_ITERATIONS iterations
      - mount is tracking (best-effort)
      - abs(HA) > 0.5 hours (i.e., <-0h30m or >+0h30m)
    Waits for Align to FINISH before returning, so Capture/Resume won't overlap slewing.
    """
    if not do_align:
        return
    if site_name != "Winter":
        return
    if ALIGN_EVERY_N_ITERATIONS <= 0:
        return
    if (iteration_idx % ALIGN_EVERY_N_ITERATIONS) != 0:
        return

    tracking, ha_hours = ekos_mount_is_tracking_and_ha_hours(ssh)
    print(f"? [{site_name}] Align check: tracking={tracking} HA_hours={ha_hours}")

    if tracking is not True:
        print(f"? [{site_name}] Align skipped: mount not confirmed tracking.")
        return
    if ha_hours is None:
        print(f"? [{site_name}] Align skipped: HA not available.")
        return
    if not (ha_hours < -ALIGN_HA_THRESHOLD_HOURS or ha_hours > ALIGN_HA_THRESHOLD_HOURS):
        print(f"? [{site_name}] Align skipped: |HA| <= {ALIGN_HA_THRESHOLD_HOURS}h.")
        return

    print(f"? [{site_name}] Aligning (capture & solve, Slew-to-Target) ...")

    # Optionally force solver action if you know the correct enum for your build.
    if FORCE_ALIGN_SOLVER_ACTION is not None:
        out, err = ssh_exec(
            ssh,
            f"qdbus org.kde.kstars /KStars/Ekos/Align org.kde.kstars.Ekos.Align.setSolverAction {int(FORCE_ALIGN_SOLVER_ACTION)}"
        )
        if out:
            print(f"? [{site_name}] Align.setSolverAction returned: {out}")
        if err:
            print(f"?? [{site_name}] Align.setSolverAction stderr: {err}")

    # Start align (async)
    out, err = ssh_exec(
        ssh,
        "qdbus org.kde.kstars /KStars/Ekos/Align org.kde.kstars.Ekos.Align.captureAndSolve"
    )
    if out:
        print(f"? [{site_name}] Align.captureAndSolve returned: {out}")
    if err:
        print(f"?? [{site_name}] Align.captureAndSolve stderr: {err}")

    # NEW: wait until Align is done BEFORE allowing Capture/Resume to proceed
    max_wait_seconds = WAIT_SECONDS * ALIGN_MAX_WAIT_SECONDS_FACTOR
    ok = wait_for_align_done(ssh, site_name, max_wait_seconds=max_wait_seconds)
    if not ok:
        print(f"?? [{site_name}] Align wait timed out; continuing anyway.")


# ==============================
# Ekos capture (default, robust)
# ==============================
def capture_ekos_once(site, iteration_idx, do_align):
    print(f"? [{site['name']}] Capturing via Ekos (robust: baseline->capture->poll->download) ...")
    ssh = None

    max_wait_seconds = WAIT_SECONDS * 3

    try:
        ssh = ssh_connect(site)

        # ---- Guiding status before ----
        st, raw, err = ekos_get_guide_status(ssh)
        if st is None:
            print(f"?? [{site['name']}] Guide status read failed. Raw: '{raw}' Err: '{err}'")
        else:
            print(f"? [{site['name']}] Guide status BEFORE suspend: {st}")

        # ---- Baseline newest FITS before capture ----
        baseline_mtime, baseline_path, baseline_err = remote_find_newest_fits_with_mtime(ssh)
        if baseline_path:
            print(f"? [{site['name']}] Baseline newest FITS: mtime={baseline_mtime:.3f} path={baseline_path}")
        else:
            print(f"?? [{site['name']}] No baseline FITS found under {REMOTE_DIR} (this is OK).")
            if baseline_err:
                print(f"?? [{site['name']}] baseline find stderr: {baseline_err}")

        # ---- Suspend guiding ----
        print(f"? [{site['name']}] Suspending guiding...")
        out, serr = ekos_guide_suspend(ssh)
        if out:
            print(f"? [{site['name']}] suspend() returned: {out}")
        if serr:
            print(f"?? [{site['name']}] suspend() stderr: {serr}")

        st2, raw2, err2 = ekos_get_guide_status(ssh)
        if st2 is None:
            print(f"?? [{site['name']}] Guide status AFTER suspend read failed. Raw: '{raw2}' Err: '{err2}'")
        else:
            print(f"? [{site['name']}] Guide status AFTER suspend: {st2}")

        # ---- NEW: Optional periodic Align (Winter only, tracking true, |HA|>0.5h), right after suspend ----
        maybe_align_winter(ssh, site["name"], iteration_idx, do_align)

        # ---- Start Ekos capture ----
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

        # ---- Robust: poll for a new FITS newer than baseline ----
        new_mtime, new_path = wait_for_new_fits(
            ssh=ssh,
            site_name=site["name"],
            baseline_mtime=baseline_mtime,
            max_wait_seconds=max_wait_seconds
        )
        if not new_path:
            return

        LAST_EKOS_REMOTE[site["name"]] = new_path

        # ---- Download to local ----
        date_str = datetime.now(UTC).replace(tzinfo=None).strftime("%Y%m%d")
        ts = datetime.now(UTC).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
        local_dir = os.path.join(LOCAL_BASE, date_str, site["name"], "guider")
        os.makedirs(local_dir, exist_ok=True)

        local_fits = os.path.join(local_dir, f"guider_{site['name']}_{ts}.fits")
        print(f"? [{site['name']}] Downloading: {new_path} -> {local_fits}")

        sftp = ssh.open_sftp()
        sftp.get(new_path, local_fits)
        sftp.close()

        print(f"? [{site['name']}] Ekos FITS saved: {local_fits}")
        convert_fits_to_png(local_fits)

    except Exception as e:
        print(f"? [{site['name']}] Ekos capture failed: {e}")

    finally:
        if ssh is not None:
            # ---- Resume guiding (best-effort) ----
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
    p.add_argument(
        "--align-every-4",
        action="store_true",
        help="If set: every N iterations (default N=4) after suspending guiding, run Ekos Align (Slew-to-Target) under constraints (Winter only, tracking true, |HA|>0h30m)."
    )
    return p.parse_args()


def main():
    args = parse_args()
    sites = load_sites(CONFIG_FILE)

    print(f"? Active sites: {[s['name'] for s in sites]}")
    print(f"? Mode: {'PHD2 (if running) else Ekos' if args.use_phd2 else 'Ekos only (default)'}")
    print(f"? Local optipng: {'FOUND' if optipng_available() else 'NOT FOUND'} (lossless PNG recompression)")
    print(f"? Robust wait: MAX_WAIT_SECONDS = WAIT_SECONDS * 3 = {WAIT_SECONDS * 3}s")
    print(f"? Periodic Align: {'ENABLED' if args.align_every_4 else 'disabled'} (every {ALIGN_EVERY_N_ITERATIONS} iterations)")

    iteration_idx = 0
    while True:
        iteration_idx += 1
        start = time.time()

        for site in sites:
            if args.use_phd2 and phd2_running_via_ssh(site):
                capture_phd2_once(site)
            else:
                capture_ekos_once(site, iteration_idx, args.align_every_4)

        elapsed = time.time() - start
        sleep_time = max(0, LOOP_INTERVAL_MIN * 60 - elapsed)
        print(f"\n? Sleeping {sleep_time/60:.1f} min...\n")
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()

