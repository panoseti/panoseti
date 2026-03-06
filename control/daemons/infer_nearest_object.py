#!/usr/bin/env python3
"""
infer_nearest_object.py

Infer the nearest *named* object to the current Ekos Mount RA/Dec (with tolerance),
and print the best match if within tolerance, otherwise print UNKNOWN.

Two modes:
  1) Live Ekos query (recommended):
        python3 infer_nearest_object.py --site Winter --tol-arcmin 10

  2) Offline / explicit coordinates:
        python3 infer_nearest_object.py --ra "05:35:17.3" --dec "-05:23:28" --tol-arcmin 5

Catalog:
  - Default: small built-in Messier subset
  - Or provide a CSV with columns: name,ra_deg,dec_deg
        python3 infer_nearest_object.py --site Winter --catalog /path/to/catalog.csv

Site resolver:
  - Reads mountcontrol.conf (CSV-ish) like:
        Winter,panoseti,panoseti-winter,5922,7624,"iOptron HAE69"
  - Default location: ./capture_mout/mountcontrol.conf (relative to current working dir)
    You can override with --mountconf.

Notes:
  - Ekos Mount.ra is typically HOURS (0..24). Mount.dec is DEGREES (-90..+90).
  - qdbus must run on the host that is running KStars/Ekos (DBus is per-user session).
  - This script queries the remote host via SSH and runs qdbus there (as the Ekos user).

Exit codes:
  0: matched within tolerance
  2: no match (UNKNOWN)
  3: runtime / query / parse error
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=3",
    "-o", "ConnectionAttempts=1",
    "-o", "StrictHostKeyChecking=accept-new",
]

DEFAULT_MOUNTCONF = os.path.join("capture_mount", "mountcontrol.conf")


# ---------------- Built-in minimal catalog ----------------
# RA/Dec in degrees (J2000-ish; sufficient for ?what object am I near?? inference)
BUILTIN_OBJECTS = [
    ("M 1",      84.03,  22.03),
    ("Mrk 421",    166.485,  38.0644),
    ("Capella",       79.66,  46.02),
    ("M45 Pleiades",        56.750000,  24.116667),
    ("M51 Whirlpool",      202.469575,  47.195258),
    ("M57 Ring Nebula",    283.396563,  33.030278),
    ("M81",                148.888221,  69.065295),
    ("M82",                148.968458,  69.679703),
    ("M104 Sombrero",      189.997917, -11.623056),
]


@dataclass
class Obj:
    name: str
    ra_deg: float
    dec_deg: float


@dataclass
class SiteCfg:
    name: str
    ssh_user: str
    ssh_host: str
    ssh_port: int


def run_cmd(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed (rc={}): {}\n{}".format(p.returncode, " ".join(cmd), (p.stderr or "").strip())
        )
    return (p.stdout or "").strip()


def parse_first_float(s: str) -> float:
    # qdbus sometimes returns additional text; grab the first float-like token.
    m = re.search(r"([-+]?\d+(?:\.\d*)?)", s)
    if not m:
        raise ValueError("Could not parse float from {!r}".format(s))
    return float(m.group(1))


def query_mount_radec_via_ssh(ssh_user: str, ssh_host: str, ssh_port: int) -> Tuple[float, float]:
    """
    Query Ekos Mount RA/Dec via SSH on the remote Ekos host.

    Returns:
      (ra_hours, dec_deg)
    """
    base = [
        "ssh", "-p", str(ssh_port),
        *SSH_OPTS,
        "{}@{}".format(ssh_user, ssh_host),
        "qdbus", "--literal",
    ]

    ra_s = run_cmd(base + ["org.kde.kstars", "/KStars/Ekos/Mount", "org.kde.kstars.Ekos.Mount.ra"])
    de_s = run_cmd(base + ["org.kde.kstars", "/KStars/Ekos/Mount", "org.kde.kstars.Ekos.Mount.dec"])

    ra_h = parse_first_float(ra_s)
    dec_d = parse_first_float(de_s)
    return ra_h, dec_d


def hms_to_hours(s: str) -> float:
    # Accept "HH:MM:SS.s" or "HH MM SS" or "HH"
    parts = re.split(r"[:\s]+", s.strip())
    parts = [p for p in parts if p]
    if len(parts) == 1:
        return float(parts[0])
    hh = float(parts[0])
    mm = float(parts[1])
    ss = float(parts[2]) if len(parts) > 2 else 0.0
    return hh + mm / 60.0 + ss / 3600.0


def dms_to_degrees(s: str) -> float:
    # Accept "+DD:MM:SS" / "-DD MM SS" / "-DD"
    s = s.strip()
    sign = -1.0 if s.startswith("-") else 1.0
    s2 = s.lstrip("+-")
    parts = re.split(r"[:\s]+", s2)
    parts = [p for p in parts if p]
    if len(parts) == 1:
        return sign * float(parts[0])
    dd = float(parts[0])
    mm = float(parts[1])
    ss = float(parts[2]) if len(parts) > 2 else 0.0
    return sign * (dd + mm / 60.0 + ss / 3600.0)


def load_catalog(path: Optional[str]) -> List[Obj]:
    if not path:
        return [Obj(n, ra, de) for (n, ra, de) in BUILTIN_OBJECTS]

    objs: List[Obj] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"name", "ra_deg", "dec_deg"}
        fields = set(reader.fieldnames or [])
        if not required.issubset(fields):
            raise ValueError(
                "Catalog must have columns {}. Found: {}".format(sorted(required), reader.fieldnames)
            )
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            objs.append(Obj(name=name, ra_deg=float(row["ra_deg"]), dec_deg=float(row["dec_deg"])))

    if not objs:
        raise ValueError("Catalog loaded zero objects from {}".format(path))
    return objs


def ang_sep_deg(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    # Spherical law of cosines (robust enough for this usage)
    ra1 = math.radians(ra1_deg)
    de1 = math.radians(dec1_deg)
    ra2 = math.radians(ra2_deg)
    de2 = math.radians(dec2_deg)
    cosd = (math.sin(de1) * math.sin(de2) +
            math.cos(de1) * math.cos(de2) * math.cos(ra1 - ra2))
    cosd = max(-1.0, min(1.0, cosd))
    return math.degrees(math.acos(cosd))


def find_nearest(objs: List[Obj], ra_deg: float, dec_deg: float) -> Tuple[Obj, float]:
    """
    Returns:
      (nearest_object, separation_deg)

    Uses astropy if available; otherwise pure-math fallback.
    """
    try:
        from astropy.coordinates import SkyCoord  # type: ignore
        import astropy.units as u  # type: ignore

        target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        cat = SkyCoord(
            ra=[o.ra_deg for o in objs] * u.deg,
            dec=[o.dec_deg for o in objs] * u.deg,
            frame="icrs",
        )
        idx, sep, _ = target.match_to_catalog_sky(cat)
        nearest = objs[int(idx)]
        return nearest, float(sep.deg)
    except Exception:
        best_o = objs[0]
        best_sep = ang_sep_deg(ra_deg, dec_deg, best_o.ra_deg, best_o.dec_deg)
        for o in objs[1:]:
            s = ang_sep_deg(ra_deg, dec_deg, o.ra_deg, o.dec_deg)
            if s < best_sep:
                best_sep = s
                best_o = o
        return best_o, best_sep


def load_sites_from_mountconf(path: str) -> Dict[str, SiteCfg]:
    """
    Parse mountcontrol.conf formatted as CSV lines, comments allowed:

      # name,ssh_user,ssh_host,ssh_port,indi_port,device
      Winter,panoseti,panoseti-winter,5922,7624,"iOptron HAE69"

    Returns dict keyed by site name (as written).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"mountconf not found: {path}")

    sites: Dict[str, SiteCfg] = {}
    with open(path, "r", newline="") as f:
        # csv.reader handles quoted device field etc.
        reader = csv.reader(f, skipinitialspace=True)
        for raw in reader:
            if not raw:
                continue
            if raw[0].strip().startswith("#"):
                continue

            # Expect at least 4 columns: name, ssh_user, ssh_host, ssh_port, ...
            if len(raw) < 4:
                continue

            name = raw[0].strip()
            ssh_user = raw[1].strip()
            ssh_host = raw[2].strip()
            try:
                ssh_port = int(str(raw[3]).strip())
            except Exception:
                raise ValueError(f"Invalid ssh_port for site {name!r} in {path}: {raw[3]!r}")

            if name:
                sites[name] = SiteCfg(name=name, ssh_user=ssh_user, ssh_host=ssh_host, ssh_port=ssh_port)

    if not sites:
        raise ValueError(f"No sites parsed from {path}")
    return sites


def main() -> int:
    ap = argparse.ArgumentParser(description="Infer nearest named object to mount RA/Dec (with tolerance).")

    ap.add_argument(
        "--mountconf",
        default=DEFAULT_MOUNTCONF,
        help=f"Path to mountcontrol.conf (default: {DEFAULT_MOUNTCONF})",
    )

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--site", help="Site name as defined in mountcontrol.conf (e.g., Winter).")
    src.add_argument("--ra", help="RA (HH:MM:SS or hours as float). Requires --dec.")

    ap.add_argument("--dec", help="Dec (+DD:MM:SS or degrees as float). Used with --ra.")
    ap.add_argument("--catalog", help="CSV catalog with columns: name,ra_deg,dec_deg.")
    ap.add_argument("--tol-arcmin", type=float, default=10.0, help="Tolerance to accept a match (arcmin).")
    ap.add_argument("--json", action="store_true", help="Also print JSON (second line).")
    ap.add_argument("--print-radec", action="store_true", help="Also print computed RA/Dec degrees.")
    args = ap.parse_args()

    try:
        # ---- Acquire RA/Dec ----
        if args.site:
            sites = load_sites_from_mountconf(args.mountconf)
            if args.site not in sites:
                known = ", ".join(sorted(sites.keys()))
                raise ValueError(f"Unknown site {args.site!r}. Known sites from {args.mountconf}: {known}")

            scfg = sites[args.site]
            ra_h, dec_d = query_mount_radec_via_ssh(scfg.ssh_user, scfg.ssh_host, scfg.ssh_port)
            ra_deg = (ra_h % 24.0) * 15.0
            dec_deg = dec_d
            source = f"ekos-mount@{args.site}"
        else:
            if args.dec is None:
                ap.error("--ra requires --dec")
            ra_h = hms_to_hours(args.ra)
            ra_deg = (ra_h % 24.0) * 15.0
            dec_deg = dms_to_degrees(args.dec)
            source = "cli"

        # ---- Load catalog + match ----
        objs = load_catalog(args.catalog)
        nearest, sep_deg = find_nearest(objs, ra_deg, dec_deg)
        sep_arcmin = sep_deg * 60.0

        matched = sep_arcmin <= float(args.tol_arcmin)
        name = nearest.name if matched else "UNKNOWN"

        # ---- Print human line ----
        extra = ""
        if args.print_radec:
            extra = f"  (ra_deg={ra_deg:.6f} dec_deg={dec_deg:.6f})"

        print(f"{name}  (sep={sep_arcmin:.2f} arcmin, tol={float(args.tol_arcmin):.2f} arcmin, source={source}){extra}")

        # ---- Optional JSON ----
        if args.json:
            out = {
                "matched": matched,
                "name": name,
                "nearest_candidate": nearest.name,
                "sep_arcmin": sep_arcmin,
                "tol_arcmin": float(args.tol_arcmin),
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "source": source,
                "catalog": args.catalog or "builtin_messier_subset",
            }
            print(json.dumps(out, sort_keys=True))

        return 0 if matched else 2

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

