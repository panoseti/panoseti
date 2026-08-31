#!/usr/bin/env python3
"""
PANOSETI ? Capture mount telemetry (SSH + INDI/qdbus)

- Reads RA/DEC (indi_getprop), target (qdbus -> INDI fallback), Alt/Az (qdbus -> transform)
- Adds strict timeouts so unreachable hosts never hang the loop
- Logs to: /mnt/data11/data/palomar/L0/YYYYMMDD/<site>/mount/<site>_mount_YYYYMMDD.log
- Publishes consolidated /tmp/mounts_current.json and uploads to cylon:/home/www/current
- Writes to Redis per-mount hash (strings only), optional UDP JSON

Assumptions:
- Remote "current" dir on cylon already exists; we do NOT mkdir remotely.
"""

import os, csv, json, time, socket, datetime, subprocess
from typing import Tuple, Optional, Dict, Any

import redis
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, FK5
from astropy.time import Time
import astropy.units as u

# ===================== CONFIG =====================
CONFIG_FILE   = "/home/obs/panoseti_mount/panoseti/control/src/control/daemons/capture_mount/mounts.conf"
SAVE_ROOT     = "/mnt/data11/data/palomar/L0"
INTERVAL_SEC  = 3.0

# Observatory location (Palomar)
SITE = EarthLocation.from_geodetic(lon="-116d51m44s", lat="33d21m12s", height=1700)

# Redis
TO_REDIS   = True
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB   = 0

# UDP (set UDP_PORT=0 to disable)
UDP_HOST = "127.0.0.1"
UDP_PORT = 5000

# Upload to cylon
REMOTE_SERVER   = "panoseti@132.239.146.24"
REMOTE_PATH     = "/web/panoseti-palomar/current/mounts_current.json"
LOCAL_TMP_JSON  = "/tmp/mounts_current.json"
BANDWIDTH_LIMIT = 40000  # kbit/s for scp (0 disables limit)

# SSH behavior
SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=3",
    "-o", "ConnectionAttempts=1",
    "-o", "StrictHostKeyChecking=accept-new",
]

SSH_CMD_TIMEOUT = 4  # seconds (per remote command)
SCP_TIMEOUT     = 4  # seconds

# ===================== HELPERS =====================
def run_ssh(ssh_user: str, ssh_host: str, ssh_port: int, remote_argv: list, timeout: int = SSH_CMD_TIMEOUT) -> Optional[str]:
    """Run a remote command via SSH; return stdout or None on error/timeout."""
    try:
        cmd = ["ssh", "-p", str(ssh_port), *SSH_OPTS, f"{ssh_user}@{ssh_host}", *remote_argv]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        if p.returncode != 0:
            return None
        return (p.stdout or "").strip()
    except Exception:
        return None

def indi_getprop(ssh_user: str, ssh_host: str, ssh_port: int, indi_port: int, prop: str) -> Optional[str]:
    """Fetch an INDI property via indi_getprop over SSH; returns value (string after '=') or None."""
    out = run_ssh(ssh_user, ssh_host, ssh_port, ["indi_getprop", "-p", str(indi_port), prop])
    if not out:
        return None
    return out.split("=", 1)[-1].strip() if "=" in out else out.strip()

def qdbus_target(ssh_user: str, ssh_host: str, ssh_port: int) -> str:
    """Get Ekos Capture target name via qdbus (empty string if unavailable)."""
    out = run_ssh(
        ssh_user, ssh_host, ssh_port,
        ["qdbus", "org.kde.kstars", "/KStars/Ekos/Capture", "org.kde.kstars.Ekos.Capture.targetName"]
    )
    return out if out else ""

def qdbus_altaz(ssh_user: str, ssh_host: str, ssh_port: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Get Alt/Az via qdbus --literal.
    Expected string contains "{az, alt}" or "{alt, az}". We try to parse two floats.
    Return (alt_deg, az_deg) or (None, None).
    """
    out = run_ssh(
        ssh_user, ssh_host, ssh_port,
        ["qdbus", "--literal", "org.kde.kstars", "/KStars/Ekos/Mount",
         "org.freedesktop.DBus.Properties.Get", "org.kde.kstars.Ekos.Mount", "horizontalCoords"]
    )
    if not out or "{" not in out or "}" not in out:
        return None, None
    try:
        inside = out.split("{", 1)[1].split("}", 1)[0]
        parts = [p.strip() for p in inside.split(",") if p.strip()]
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                pass
        if len(nums) < 2:
            return None, None
        # Heuristic: many versions return {Az, Alt}. Sanity-check Alt range.
        az, alt = nums[0], nums[1]
        if not (-10 <= alt <= 90) and (-10 <= az <= 90):
            alt, az = az, az  # fallback unlikely; keep original if nonsense
        return float(alt), float(az)
    except Exception:
        return None, None

def parse_ra_hours(val: str) -> Optional[float]:
    """RA string -> decimal hours. Accepts '2.5' or '02:30:00.0'."""
    if val is None:
        return None
    s = val.strip()
    try:
        if ":" in s:
            h, m, sec = [float(x) for x in s.split(":")]
            return h + m/60 + sec/3600
        return float(s)
    except Exception:
        return None

def parse_dec_deg(val: str) -> Optional[float]:
    """Dec string -> decimal degrees. Accepts '-12.5' or '-12:30:00.0' (with +/?)."""
    if val is None:
        return None
    s = val.strip()
    try:
        if ":" in s:
            sign = -1 if s.startswith("-") else 1
            s2 = s.replace("+", "").replace("-", "")
            d, m, sec = [float(x) for x in s2.split(":")]
            return sign * (d + m/60 + sec/3600)
        return float(s)
    except Exception:
        return None

def ensure_day_log(name: str) -> str:
    day = datetime.datetime.utcnow().strftime("%Y%m%d")
    path = os.path.join(SAVE_ROOT, day, name, "mount")
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, f"{name}_mount_{day}.log")

def write_log(logfile: str, line: str) -> None:
    try:
        with open(logfile, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

def safe_redis_mapping(d: Dict[str, Any]) -> Dict[str, str]:
    """Convert values to strings; skip None (Redis cannot store None)."""
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, bool):
            out[k] = "1" if v else "0"
        else:
            out[k] = str(v)
    return out

def load_mounts(conf_path: str):
    mounts = []
    try:
        with open(conf_path) as f:
            for row in csv.reader(f):
                if not row or row[0].strip().startswith("#"):
                    continue
                try:
                    n, u, h, sp, ip, d = row
                    mounts.append(dict(
                        name=n.strip(),
                        ssh_user=u.strip(),
                        ssh_host=h.strip(),
                        ssh_port=int(sp),
                        indi_port=int(ip),
                        device=d.strip().strip('"'),
                    ))
                except Exception:
                    print("[config] skip row:", row)
    except FileNotFoundError:
        print(f"[ERROR] config not found: {conf_path}")
    return mounts

def scp_upload(local_path: str, remote_server: str, remote_path: str) -> None:
    """Upload file with scp, bandwidth limit and timeout; never raises."""
    cmd = ["scp", *SSH_OPTS]
    if BANDWIDTH_LIMIT and BANDWIDTH_LIMIT > 0:
        cmd += ["-l", str(BANDWIDTH_LIMIT)]
    cmd += [local_path, f"{remote_server}:{remote_path}"]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=SCP_TIMEOUT, check=False)
    except Exception:
        pass

# ===================== MAIN =====================
def main():
    mounts = load_mounts(CONFIG_FILE)
    if not mounts:
        print("[capture_mounts] No mounts configured.")
        return

    print(f"[capture_mounts] Loaded {len(mounts)} mounts")

    # Redis (optional)
    r = None
    if TO_REDIS:
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
            r.ping()
        except Exception as e:
            print(f"[WARN] Redis unavailable: {e}")
            r = None

    # UDP (optional)
    udp_sock = None
    if UDP_PORT:
        try:
            udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except Exception as e:
            print(f"[WARN] UDP disabled: {e}")
            udp_sock = None

    # Prepare log file map (refresh if day rolls)
    current_day = None
    log_paths: Dict[str, str] = {}

    while True:
        now = Time.now()
        iso = now.isot
        day = now.datetime.strftime("%Y%m%d")
        if day != current_day:
            current_day = day
            log_paths = {m["name"]: ensure_day_log(m["name"]) for m in mounts}
            for name, lf in log_paths.items():
                print(f"[LOG] {name} -> {lf}")

        combined = {"timestamp": iso, "mounts": {}}

        for m in mounts:
            name      = m["name"]
            ssh_user  = m["ssh_user"]
            ssh_host  = m["ssh_host"]
            ssh_port  = m["ssh_port"]
            indi_port = m["indi_port"]
            dev       = m["device"]
            logf      = log_paths[name]

            # ---------- Query INDI RA/DEC ----------
            ra_str  = indi_getprop(ssh_user, ssh_host, ssh_port, indi_port, f'"{dev}.EQUATORIAL_EOD_COORD.RA"')
            dec_str = indi_getprop(ssh_user, ssh_host, ssh_port, indi_port, f'"{dev}.EQUATORIAL_EOD_COORD.DEC"')

            if not ra_str or not dec_str:
                msg = f"[{name}] {iso} | NO DATA (mount offline or INDI unreachable)"
                print(msg); write_log(logf, msg)
                combined["mounts"][name] = {
                    "timestamp": iso, "mount": name, "status": "OFFLINE",
                    "ra_hours": None, "dec_deg": None, "alt_deg": None, "az_deg": None,
                    "ra_hours_j2000": None, "dec_deg_j2000": None,
                    "side": None, "side_code": None, "tracking": None, "tracking_code": None,
                    "target_name": ""
                }
                # still update Redis to reflect offline state
                if r:
                    r.hset(f"MOUNT_{name.upper()}", mapping={"timestamp": iso, "status": "OFFLINE"})
                continue

            ra_h  = parse_ra_hours(ra_str)
            dec_d = parse_dec_deg(dec_str)
            if ra_h is None or dec_d is None:
                msg = f"[{name}] {iso} | BAD RA/DEC FORMAT ({ra_str}, {dec_str})"
                print(msg); write_log(logf, msg)
                # record placeholder so UI still shows a panel
                snap = {
                    "timestamp": iso, "mount": name, "status": "BAD_RADEC",
                    "ra_hours": None, "dec_deg": None, "alt_deg": None, "az_deg": None,
                    "ra_hours_j2000": None, "dec_deg_j2000": None,
                    "side": None, "side_code": None, "tracking": None, "tracking_code": None,
                    "target_name": qdbus_target(ssh_user, ssh_host, ssh_port) or ""
                }
                combined["mounts"][name] = snap
                if r: r.hset(f"MOUNT_{name.upper()}", mapping=safe_redis_mapping(snap))
                continue

            # ---------- Compute J2000 RA/Dec from EOD ----------
            # NOTE: Mount reports EQUATORIAL_EOD_COORD (of-date). Convert to FK5(J2000).
            ra_j2000_h, dec_j2000_d = None, None
            try:
                sc_eod = SkyCoord(ra=ra_h*u.hourangle, dec=dec_d*u.deg, frame=FK5(equinox=now))
                sc_j2000 = sc_eod.transform_to(FK5(equinox=Time("J2000")))
                ra_j2000_h = float(sc_j2000.ra.hour)
                dec_j2000_d = float(sc_j2000.dec.deg)
            except Exception:
                ra_j2000_h, dec_j2000_d = None, None

            # ---------- AltAz via qdbus, else transform ----------
            alt_q, az_q = qdbus_altaz(ssh_user, ssh_host, ssh_port)
            if alt_q is not None and az_q is not None:
                alt_deg, az_deg = float(alt_q), float(az_q)
            else:
                sc = SkyCoord(ra=ra_h*u.hourangle, dec=dec_d*u.deg, frame="icrs")
                altaz = sc.transform_to(AltAz(obstime=now, location=SITE))
                alt_deg, az_deg = float(altaz.alt.deg), float(altaz.az.deg)

            # ---------- Other telemetry ----------
            target = qdbus_target(ssh_user, ssh_host, ssh_port)
            if not target:
                target = indi_getprop(ssh_user, ssh_host, ssh_port, indi_port, f'"{dev}.TELESCOPE_TARGET_NAME.NAME"') or ""

            pw = indi_getprop(ssh_user, ssh_host, ssh_port, indi_port, f'"{dev}.TELESCOPE_PIER_SIDE.PIER_WEST"')
            pe = indi_getprop(ssh_user, ssh_host, ssh_port, indi_port, f'"{dev}.TELESCOPE_PIER_SIDE.PIER_EAST"')
            side = "WEST" if pw == "On" else ("EAST" if pe == "On" else None)
            side_code = 1 if side == "WEST" else (0 if side == "EAST" else None)

            t_on = indi_getprop(ssh_user, ssh_host, ssh_port, indi_port, f'"{dev}.TELESCOPE_TRACK_STATE.TRACK_ON"')
            tracking = (t_on == "On")
            tracking_code = 1 if tracking else 0

            # ---------- Log line ----------
            line = (f"[{name}] {iso} | RA:{ra_h:.6f}h Dec:{dec_d:.6f}° "
                    f"Alt:{alt_deg:.2f}° Az:{az_deg:.2f}° "
                    f"RA_J2000:{(ra_j2000_h if ra_j2000_h is not None else float('nan')):.6f}h "
                    f"Dec_J2000:{(dec_j2000_d if dec_j2000_d is not None else float('nan')):.6f}° "
                    f"Side:{side}({side_code}) "
                    f"Tracking:{tracking}({tracking_code}) Target:{target}")
            print(line); write_log(logf, line)

            # ---------- Snapshot ----------
            snap = dict(
                timestamp=iso,
                computer_utc=time.time(),
                mount=name,
                status="OK",
                ra_hours=round(ra_h, 6),
                dec_deg=round(dec_d, 6),
                alt_deg=round(alt_deg, 3),
                az_deg=round(az_deg, 3),
                ra_hours_j2000=(round(ra_j2000_h, 6) if ra_j2000_h is not None else None),
                dec_deg_j2000=(round(dec_j2000_d, 6) if dec_j2000_d is not None else None),
                side=side,
                side_code=side_code,
                tracking=tracking,
                tracking_code=tracking_code,
                target_name=target
            )
            combined["mounts"][name] = snap

            # Redis
            if r:
                try:
                    r.hset(f"MOUNT_{name.upper()}", mapping=safe_redis_mapping(snap))
                except Exception as e:
                    print(f"[WARN] Redis HSET failed for {name}: {e}")

            # UDP
            if udp_sock and UDP_PORT:
                try:
                    udp_sock.sendto(json.dumps(snap).encode("utf-8"), (UDP_HOST, UDP_PORT))
                except Exception:
                    pass

        # ---------- Write & upload mounts_current.json ----------
        try:
            with open(LOCAL_TMP_JSON, "w") as f:
                json.dump(combined, f, indent=2)
        except Exception as e:
            print(f"[WARN] cannot write {LOCAL_TMP_JSON}: {e}")

        scp_upload(LOCAL_TMP_JSON, REMOTE_SERVER, REMOTE_PATH)

        time.sleep(INTERVAL_SEC)

# ===================== ENTRY =====================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.\n")

