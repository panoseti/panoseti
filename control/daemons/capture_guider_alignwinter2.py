#!/usr/bin/env python3
"""
capture_guider.py

Periodically capture guider images from multiple remote sites (SSH/scp).

Modes:
  - Default: Ekos Capture (robust: baseline newest FITS -> start capture -> wait for newer FITS -> download)
  - Optional: PHD2 get_star_image (if running), otherwise fallback to Ekos.

Optional periodic Align (Winter only):
  - Triggered every ALIGN_EVERY_N_ITERATIONS outer-loop iterations
  - Only if mount status indicates TRACKING (Mount.status == 3)
  - Only if |hourAngle| > ALIGN_HA_THRESHOLD_HOURS

NEW (Winter only, when Align is due):
  - After guiding is suspended (and before capture), infer the nearest named target
    from the current mount coordinates (Mount.equatorialCoords) with a tolerance.
  - If a target is inferred, run the SAME command-style sequence as mount_target.py, using
    scripts in /home/obs/panoseti_mount/panoseti/test/:
      1) python3 /home/obs/panoseti_mount/panoseti/test/kstars_align_slew.py --site Winter --target "<name>" --action sync
      2) python3 /home/obs/panoseti_mount/panoseti/test/kstars_goto.py       --site Winter --target "<name>" --wait
      3) python3 /home/obs/panoseti_mount/panoseti/test/ekos_guiding.py      --site Winter start

Notes:
  - No paramiko dependency.
  - Uses mountcontrol.conf for site resolution (no SSH args exposed).
  - Guiding is suspended via direct DBus call (Guide.suspend) before the sequence.
    Guiding is restarted via ekos_guiding.py start (same as your workflow).

Tested for Python 3.8+.
"""

import argparse
import base64
import csv
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


# ---------------- CONFIG ----------------
MOUNTCONF = "/home/obs/panoseti_mount/panoseti/control/daemons/capture_mount/mountcontrol.conf"

REMOTE_DIR = "/home/panoseti/Pictures"
TRAIN_NAME = "Primary"

WAIT_SECONDS = 15
LOOP_INTERVAL_MIN = 3
LOCAL_BASE = "/mnt/data11/data/palomar/L0"

OPTIPNG_LEVEL = 5  # 0..7 (lossless). 5 is a good speed/ratio balance.

SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=3",
    "-o", "ConnectionAttempts=1",
    "-o", "StrictHostKeyChecking=accept-new",
]
SSH_TIMEOUT = 20

# --- Optional periodic Align (Winter only) ---
ALIGN_EVERY_N_ITERATIONS = 2          # do align every N outer-loop iterations
ALIGN_HA_THRESHOLD_HOURS = 0.2        # e.g. 0.2h = 12 minutes
# ----------------------------------------

# --- Target inference settings (Winter only, when Align is due) ---
INFER_TARGET_TOL_ARCMIN = 40.0        # accept nearest target only within this tolerance
TARGET_CATALOG_CSV = None             # optional CSV catalog: columns name,ra_deg,dec_deg
# ----------------------------------------------------------------

# --- Helper scripts (fixed directory per your note) ---
HELPER_DIR = "/home/obs/panoseti_mount/panoseti/test"
PYTHON_BIN = "python3"
KSTARS_GOTO = os.path.join(HELPER_DIR, "kstars_goto.py")
KSTARS_ALIGN_SLEW = os.path.join(HELPER_DIR, "kstars_align_slew.py")
EKOS_GUIDING = os.path.join(HELPER_DIR, "ekos_guiding.py")
# ------------------------------------------------------

# Tracks last-downloaded remote file per site (extra safety against repeats)
LAST_EKOS_REMOTE: Dict[str, str] = {}  # site_name -> remote_path


# ==============================
# Site config (mountcontrol.conf)
# ==============================
@dataclass(frozen=True)
class SiteConf:
    name: str
    ssh_user: str
    ssh_host: str
    ssh_port: int
    indi_port: int
    device: str


def load_mountconf(conf_path: str) -> Dict[str, SiteConf]:
    out: Dict[str, SiteConf] = {}
    with open(conf_path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 6:
                continue
            name, user, host, ssh_port, indi_port, device = [x.strip() for x in row[:6]]
            out[name.lower()] = SiteConf(
                name=name,
                ssh_user=user,
                ssh_host=host,
                ssh_port=int(ssh_port),
                indi_port=int(indi_port),
                device=device.strip().strip('"'),
            )
    return out


def run_ssh(site: SiteConf, remote_cmd: str, timeout: int = SSH_TIMEOUT) -> str:
    cmd = [
        "ssh", "-p", str(site.ssh_port),
        *SSH_OPTS,
        f"{site.ssh_user}@{site.ssh_host}",
        remote_cmd,
    ]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError("SSH timeout")
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip() or f"ssh rc={p.returncode}")
    return (p.stdout or "").strip()


def run_scp_get(site: SiteConf, remote_path: str, local_path: str, timeout: int = 60) -> None:
    cmd = [
        "scp", "-P", str(site.ssh_port),
        f"{site.ssh_user}@{site.ssh_host}:{remote_path}",
        local_path,
    ]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError("SCP timeout")
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip() or f"scp rc={p.returncode}")


# ==============================
# Target inference (nearest object)
# ==============================
@dataclass
class Obj:
    name: str
    ra_deg: float
    dec_deg: float


BUILTIN_OBJECTS = [
    Obj("M 1",      84.03,   22.03),
    Obj("Mrk 421",  166.485, 38.0644),
    Obj("Capella",  79.66,   46.02),
    Obj("M42 Orion Nebula", 83.822083,  -5.391111),
    Obj("M45 Pleiades",     56.750000,  24.116667),
    Obj("M51 Whirlpool",   202.469575,  47.195258),
    Obj("M57 Ring Nebula", 283.396563,  33.030278),
    Obj("M81",             148.888221,  69.065295),
    Obj("M82",             148.968458,  69.679703),
    Obj("M104 Sombrero",   189.997917, -11.623056),
]


def _angular_sep_deg(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    ra1 = math.radians(ra1_deg)
    de1 = math.radians(dec1_deg)
    ra2 = math.radians(ra2_deg)
    de2 = math.radians(dec2_deg)
    cosd = (math.sin(de1) * math.sin(de2) +
            math.cos(de1) * math.cos(de2) * math.cos(ra1 - ra2))
    cosd = max(-1.0, min(1.0, cosd))
    return math.degrees(math.acos(cosd))


def _load_target_catalog(csv_path: Optional[str]) -> List[Obj]:
    if not csv_path:
        return list(BUILTIN_OBJECTS)

    objs: List[Obj] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"name", "ra_deg", "dec_deg"}
        fields = set(reader.fieldnames or [])
        if not required.issubset(fields):
            raise ValueError(f"Target catalog must have columns {sorted(required)}. Found: {reader.fieldnames}")
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            objs.append(Obj(name=name, ra_deg=float(row["ra_deg"]), dec_deg=float(row["dec_deg"])))

    if not objs:
        raise ValueError(f"Target catalog loaded zero objects from {csv_path}")
    return objs


def _find_nearest_target(objs: List[Obj], ra_deg: float, dec_deg: float) -> Tuple[Obj, float]:
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        cat = SkyCoord(
            ra=[o.ra_deg for o in objs] * u.deg,
            dec=[o.dec_deg for o in objs] * u.deg,
            frame="icrs",
        )
        idx, sep, _ = target.match_to_catalog_sky(cat)
        return objs[int(idx)], float(sep.deg)
    except Exception:
        best = objs[0]
        best_sep = _angular_sep_deg(ra_deg, dec_deg, best.ra_deg, best.dec_deg)
        for o in objs[1:]:
            s = _angular_sep_deg(ra_deg, dec_deg, o.ra_deg, o.dec_deg)
            if s < best_sep:
                best = o
                best_sep = s
        return best, best_sep


def ekos_mount_equatorial_coords_deg(site: SiteConf) -> Tuple[float, float]:
    cmd = (
        "qdbus --literal org.kde.kstars /KStars/Ekos/Mount "
        "org.freedesktop.DBus.Properties.Get org.kde.kstars.Ekos.Mount equatorialCoords"
    )
    out = run_ssh(site, cmd)
    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", out or "")]
    if len(nums) < 2:
        raise RuntimeError(f"Could not parse equatorialCoords from: {out!r}")

    ra_val = nums[0]
    dec_deg = nums[1]

    if ra_val > 24.0:
        ra_deg = ra_val % 360.0
    else:
        ra_deg = (ra_val % 24.0) * 15.0

    return ra_deg, dec_deg


def infer_current_target_name(site: SiteConf, tol_arcmin: float, catalog_csv: Optional[str]) -> Tuple[str, float, float, float]:
    objs = _load_target_catalog(catalog_csv)
    ra_deg, dec_deg = ekos_mount_equatorial_coords_deg(site)
    nearest, sep_deg = _find_nearest_target(objs, ra_deg, dec_deg)
    sep_arcmin = sep_deg * 60.0

    if sep_arcmin <= float(tol_arcmin):
        return nearest.name, sep_arcmin, ra_deg, dec_deg
    return "UNKNOWN", sep_arcmin, ra_deg, dec_deg


# ==============================
# Misc helpers
# ==============================
def utc_datestr():
    return datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y%m%d")


def utc_ts_compact():
    return datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")


def site_norm(name: str) -> str:
    return (name or "").strip().lower()


def extract_last_int(s: str) -> Optional[int]:
    nums = re.findall(r"-?\d+", s or "")
    if not nums:
        return None
    try:
        return int(nums[-1])
    except Exception:
        return None


def extract_last_float(s: str) -> Optional[float]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", s or "")
    if not nums:
        return None
    try:
        return float(nums[-1])
    except Exception:
        return None


def run_local(argv: List[str]) -> int:
    return subprocess.run(argv).returncode


# ==============================
# PNG lossless recompression
# ==============================
def optipng_available():
    return shutil.which("optipng") is not None


def recompress_png_lossless(png_path, level=OPTIPNG_LEVEL):
    if not optipng_available():
        print(f"?? optipng not found, skipping PNG recompression: {png_path}")
        return

    try:
        size_before = os.path.getsize(png_path)
        subprocess.run(
            ["optipng", f"-o{level}", png_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        size_after = os.path.getsize(png_path)
        if size_after < size_before:
            print(f"? PNG recompressed (lossless): {png_path} ({size_before/1024:.1f} KB -> {size_after/1024:.1f} KB)")
        else:
            print(f"? PNG recompressed (lossless): {png_path} (no gain)")
    except Exception as e:
        print(f"?? optipng failed for {png_path}: {e}")


def convert_fits_to_png(fits_path):
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
        print(f"?? FITS conversion failed for {fits_path}: {e}")


# ==============================
# Ekos helpers (remote via ssh)
# ==============================
def ekos_get_guide_status(site: SiteConf):
    cmd = (
        "qdbus org.kde.kstars /KStars/Ekos/Guide "
        "org.freedesktop.DBus.Properties.Get org.kde.kstars.Ekos.Guide status"
    )
    out = run_ssh(site, cmd)
    st = extract_last_int(out)
    return st, out


def ekos_guide_suspend(site: SiteConf):
    cmd = "qdbus org.kde.kstars /KStars/Ekos/Guide org.kde.kstars.Ekos.Guide.suspend"
    return run_ssh(site, cmd)


def ekos_mount_tracking_and_ha_hours(site: SiteConf) -> Tuple[Optional[bool], Optional[float]]:
    tracking = None
    ha_hours = None

    cmd_status = (
        "qdbus org.kde.kstars /KStars/Ekos/Mount "
        "org.freedesktop.DBus.Properties.Get org.kde.kstars.Ekos.Mount status"
    )
    out = run_ssh(site, cmd_status)
    st = extract_last_int(out)
    if st is not None:
        tracking = (st == 3)

    cmd_ha = (
        "qdbus org.kde.kstars /KStars/Ekos/Mount "
        "org.freedesktop.DBus.Properties.Get org.kde.kstars.Ekos.Mount hourAngle"
    )
    out2 = run_ssh(site, cmd_ha)
    ha_hours = extract_last_float(out2)

    return tracking, ha_hours


def remote_find_newest_fits_with_mtime(site: SiteConf):
    cmd = (
        f'find {REMOTE_DIR} -type f -name "*.fit*" '
        '-printf "%T@ %p\\n" | sort -nr | head -n 1'
    )
    out = run_ssh(site, cmd)
    if not out:
        return None, "", ""

    parts = out.split(" ", 1)
    if len(parts) != 2:
        return None, "", ""

    try:
        mtime = float(parts[0])
    except Exception:
        mtime = None

    path = parts[1].strip()
    return mtime, path, ""


def wait_for_new_fits(site: SiteConf, site_name: str, baseline_mtime, max_wait_seconds):
    poll_interval = 1.0
    deadline = time.time() + max_wait_seconds
    last_seen = LAST_EKOS_REMOTE.get(site_name)

    print(f"? [{site_name}] Waiting for new FITS (baseline mtime={baseline_mtime}) max_wait={max_wait_seconds}s ...")

    while time.time() < deadline:
        mtime, path, _ = remote_find_newest_fits_with_mtime(site)
        if path and (mtime is not None):
            if (baseline_mtime is None or mtime > baseline_mtime) and (path != last_seen):
                print(f"? [{site_name}] New FITS detected: mtime={mtime:.3f} path={path}")
                return mtime, path
        time.sleep(poll_interval)

    print(f"?? [{site_name}] Timed out waiting for a new FITS.")
    return None, ""


# ==============================
# Winter due-align sequence via your existing helper scripts
# ==============================
def run_align_sync_goto_and_restart_guiding(site_name: str, target_name: str) -> int:
    """
    Use the same command pattern as mount_target.py for:
      - align sync (kstars_align_slew.py --action sync --target "<name>")
      - goto wait (kstars_goto.py --wait --target "<name>")
      - restart guiding (ekos_guiding.py start)

    Returns rc != 0 if any step fails (stops immediately).
    """
    # 1) align sync
    rc = run_local([PYTHON_BIN, KSTARS_ALIGN_SLEW, "--site", site_name, "--target", target_name, "--action", "sync"])
    if rc != 0:
        return rc

    # 2) goto wait
    rc = run_local([PYTHON_BIN, KSTARS_GOTO, "--site", site_name, "--target", target_name, "--wait"])
    if rc != 0:
        return rc

    # 3) start guiding
    rc = run_local([PYTHON_BIN, EKOS_GUIDING, "--site", site_name, "start"])
    return rc


def maybe_align_winter(site: SiteConf, iteration_idx: int, do_align: bool):
    if not do_align:
        return
    if site_norm(site.name) != "winter":
        return
    if ALIGN_EVERY_N_ITERATIONS <= 0:
        print(f"? [{site.name}] Align disabled: ALIGN_EVERY_N_ITERATIONS={ALIGN_EVERY_N_ITERATIONS}")
        return

    due = (iteration_idx % ALIGN_EVERY_N_ITERATIONS) == 0
    print(f"? [{site.name}] Align schedule: iter={iteration_idx} everyN={ALIGN_EVERY_N_ITERATIONS} due={due}")
    if not due:
        return

    tracking, ha_hours = ekos_mount_tracking_and_ha_hours(site)
    print(f"? [{site.name}] Align gate: tracking={tracking} HA_hours={ha_hours} threshold={ALIGN_HA_THRESHOLD_HOURS}h")

    if tracking is not True:
        print(f"? [{site.name}] Align skipped: mount not confirmed TRACKING (status!=3 or unreadable).")
        return
    if ha_hours is None:
        print(f"? [{site.name}] Align skipped: hourAngle unavailable/unreadable.")
        return
    if not (ha_hours < -ALIGN_HA_THRESHOLD_HOURS or ha_hours > ALIGN_HA_THRESHOLD_HOURS):
        print(f"? [{site.name}] Align skipped: |HA| <= {ALIGN_HA_THRESHOLD_HOURS}h.")
        return

    # infer target
    try:
        inferred, sep_arcmin, ra_deg, dec_deg = infer_current_target_name(
            site=site,
            tol_arcmin=INFER_TARGET_TOL_ARCMIN,
            catalog_csv=TARGET_CATALOG_CSV,
        )
        print(
            f"? [{site.name}] Target inference: name={inferred} sep={sep_arcmin:.2f} arcmin "
            f"(tol={INFER_TARGET_TOL_ARCMIN:.2f}) radec=({ra_deg:.6f},{dec_deg:.6f})"
        )
    except Exception as e:
        inferred = "UNKNOWN"
        print(f"?? [{site.name}] Target inference failed: {e}")

    if inferred == "UNKNOWN":
        print(f"? [{site.name}] Align skipped: could not infer target within tolerance.")
        return

    print(f"? [{site.name}] Helper-script sequence on inferred target '{inferred}': align-sync + goto(wait) + guiding start ...")
    rc = run_align_sync_goto_and_restart_guiding(site.name, inferred)
    if rc != 0:
        print(f"?? [{site.name}] Helper-script sequence failed (exit={rc}).")


# ==============================
# Ekos capture
# ==============================
def capture_ekos_once(site: SiteConf, iteration_idx: int, do_align: bool):
    name = site.name
    print(f"? [{name}] Capturing via Ekos (baseline->capture->poll->download) ...")

    max_wait_seconds = WAIT_SECONDS * 3
    align_due = (do_align and site_norm(name) == "winter" and ALIGN_EVERY_N_ITERATIONS > 0 and (iteration_idx % ALIGN_EVERY_N_ITERATIONS) == 0)

    try:
        # Guide status before
        st, raw = ekos_get_guide_status(site)
        if st is None:
            print(f"?? [{name}] Guide status read failed. Raw='{raw}'")
        else:
            print(f"? [{name}] Guide status BEFORE suspend: {st}")

        # Baseline newest FITS
        baseline_mtime, baseline_path, baseline_err = remote_find_newest_fits_with_mtime(site)
        if baseline_path:
            if baseline_mtime is None:
                print(f"? [{name}] Baseline newest FITS path={baseline_path} (mtime parse failed)")
            else:
                print(f"? [{name}] Baseline newest FITS: mtime={baseline_mtime:.3f} path={baseline_path}")
        else:
            print(f"?? [{name}] No baseline FITS found under {REMOTE_DIR} (this is OK).")
            if baseline_err:
                print(f"?? [{name}] baseline find stderr: {baseline_err}")

        # Suspend guiding
        print(f"? [{name}] Suspending guiding...")
        out = ekos_guide_suspend(site)
        if out:
            print(f"? [{name}] suspend() returned: {out}")

        st2, raw2 = ekos_get_guide_status(site)
        if st2 is None:
            print(f"?? [{name}] Guide status AFTER suspend read failed. Raw='{raw2}'")
        else:
            print(f"? [{name}] Guide status AFTER suspend: {st2}")

        # Optional winter align-sync + goto(wait) + guiding start (via helper scripts)
        maybe_align_winter(site, iteration_idx, do_align)

        # Start capture
        print(f"? [{name}] Starting Ekos Capture for optical train '{TRAIN_NAME}'...")
        out_cap = run_ssh(
            site,
            "qdbus org.kde.kstars /KStars/Ekos/Capture "
            f"org.kde.kstars.Ekos.Capture.start \"{TRAIN_NAME}\""
        )
        if out_cap:
            print(f"? [{name}] Capture.start returned: {out_cap}")

        # Wait for new FITS newer than baseline
        new_mtime, new_path = wait_for_new_fits(
            site=site,
            site_name=name,
            baseline_mtime=baseline_mtime,
            max_wait_seconds=max_wait_seconds,
        )
        if not new_path:
            return

        LAST_EKOS_REMOTE[name] = new_path

        # Download via scp
        date_str = utc_datestr()
        ts = utc_ts_compact()
        local_dir = os.path.join(LOCAL_BASE, date_str, name, "guider")
        os.makedirs(local_dir, exist_ok=True)

        local_fits = os.path.join(local_dir, f"guider_{name}_{ts}.fits")
        print(f"? [{name}] Downloading: {new_path} -> {local_fits}")
        run_scp_get(site, new_path, local_fits, timeout=60)

        print(f"? [{name}] Ekos FITS saved: {local_fits}")
        convert_fits_to_png(local_fits)

    except Exception as e:
        print(f"?? [{name}] Ekos capture failed: {e}")

    finally:
        # For Winter when align_due, guiding was restarted via ekos_guiding.py start.
        # For everyone else, restart guiding using the SAME method (ekos_guiding.py start).
        try:
            if not align_due:
                print(f"? [{name}] Restart guiding via {EKOS_GUIDING} ...")
                rc = run_local([PYTHON_BIN, EKOS_GUIDING, "--site", name, "start"])
                if rc != 0:
                    print(f"?? [{name}] ekos_guiding.py start failed (exit={rc})")
        except Exception as e:
            print(f"?? [{name}] Guide restart attempt failed: {e}")


# ==============================
# PHD2 helpers (optional) via ssh
# ==============================
def phd2_running(site: SiteConf) -> bool:
    try:
        cmd = "echo '{\"method\":\"get_app_state\",\"id\":1,\"jsonrpc\":\"2.0\"}' | nc -w 2 localhost 4400"
        out = run_ssh(site, cmd, timeout=8)
        return ("PHDVersion" in out) or ('\"result\"' in out)
    except Exception:
        return False


def capture_phd2_once(site: SiteConf):
    print(f"? [{site.name}] Capturing via PHD2 ...")
    try:
        cmd = "echo '{\"method\":\"get_star_image\",\"id\":1,\"jsonrpc\":\"2.0\"}' | nc -w 5 localhost 4400"
        output = run_ssh(site, cmd, timeout=12)

        reply = None
        for line in (output or "").splitlines():
            try:
                reply = json.loads(line)
            except Exception:
                continue

        if not reply or "result" not in reply or "image" not in reply["result"]:
            print(f"?? [{site.name}] No image data from PHD2.")
            return

        img_b64 = reply["result"]["image"]
        w = int(reply["result"]["width"])
        h = int(reply["result"]["height"])
        arr = np.frombuffer(base64.b64decode(img_b64), dtype=np.uint16).reshape(h, w)

        date_str = utc_datestr()
        ts = utc_ts_compact()
        local_dir = os.path.join(LOCAL_BASE, date_str, site.name, "guider")
        os.makedirs(local_dir, exist_ok=True)

        fits_path = os.path.join(local_dir, f"guider_{site.name}_{ts}.fits")
        fits.writeto(fits_path, arr, overwrite=True)

        print(f"? [{site.name}] FITS saved: {fits_path}")
        convert_fits_to_png(fits_path)

    except Exception as e:
        print(f"?? [{site.name}] PHD2 capture failed: {e}")


# ==============================
# Main / CLI
# ==============================
def parse_args():
    p = argparse.ArgumentParser(description="Capture guider images periodically from remote sites (ssh/scp).")
    p.add_argument("--use-phd2", action="store_true",
                   help="Use PHD2 get_star_image if PHD2 is running remotely; otherwise fall back to Ekos.")
    p.add_argument("--align-every-4", action="store_true",
                   help="Enable periodic Align for Winter (every ALIGN_EVERY_N_ITERATIONS iterations) after suspending guiding.")
    p.add_argument("--mountconf", default=MOUNTCONF,
                   help=f"mountcontrol.conf path (default: {MOUNTCONF})")

    p.add_argument("--target-catalog", default=None,
                   help="Optional CSV catalog with columns name,ra_deg,dec_deg (overrides built-in subset).")
    p.add_argument("--infer-tol-arcmin", type=float, default=None,
                   help="Override inference tolerance in arcmin (default INFER_TARGET_TOL_ARCMIN).")
    return p.parse_args()


def main():
    global TARGET_CATALOG_CSV, INFER_TARGET_TOL_ARCMIN

    args = parse_args()
    conf = load_mountconf(args.mountconf)
    sites = [conf[k] for k in sorted(conf.keys())]  # stable order

    if args.target_catalog is not None:
        TARGET_CATALOG_CSV = args.target_catalog
    if args.infer_tol_arcmin is not None:
        INFER_TARGET_TOL_ARCMIN = float(args.infer_tol_arcmin)

    print(f"? Active sites: {[s.name for s in sites]}")
    print(f"? Mode: {'PHD2 (if running) else Ekos' if args.use_phd2 else 'Ekos only (default)'}")
    print(f"? Local optipng: {'FOUND' if optipng_available() else 'NOT FOUND'} (lossless PNG recompression)")
    print(f"? Robust wait: MAX_WAIT_SECONDS = WAIT_SECONDS * 3 = {WAIT_SECONDS * 3}s")
    print(f"? Periodic Align: {'ENABLED' if args.align_every_4 else 'disabled'} (every {ALIGN_EVERY_N_ITERATIONS} iterations)")
    if args.align_every_4:
        print(f"? Align gates: Winter-only, Mount.status==3, |HA|>{ALIGN_HA_THRESHOLD_HOURS}h")
        print(f"? Target inference: tol={INFER_TARGET_TOL_ARCMIN} arcmin catalog={TARGET_CATALOG_CSV or 'BUILTIN'}")
        print(f"? Helper scripts dir: {HELPER_DIR}")
        print("? Winter due-align sequence: STOP guiding -> infer -> kstars_align_slew.py sync (with --target) -> kstars_goto.py --wait (with --target) -> ekos_guiding.py start")

    iteration_idx = 0
    while True:
        iteration_idx += 1
        start = time.time()

        for site in sites:
            if args.use_phd2 and phd2_running(site):
                capture_phd2_once(site)
            else:
                capture_ekos_once(site=site, iteration_idx=iteration_idx, do_align=args.align_every_4)

        elapsed = time.time() - start
        sleep_time = max(0.0, LOOP_INTERVAL_MIN * 60.0 - elapsed)
        print(f"\n? Sleeping {sleep_time/60.0:.1f} min...\n")
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()


