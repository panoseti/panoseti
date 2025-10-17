#!/usr/bin/env python3

import argparse
import subprocess
import time
import socket
import redis
import datetime
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u

# -------------------- CONFIG --------------------
SITE_LAT = "33d21m12s"
SITE_LON = "-116d51m44s"
SITE_ELEV = 1700  # meters, adjust as needed
SITE = EarthLocation.from_geodetic(SITE_LON, SITE_LAT, SITE_ELEV)

# -------------------- UTILS --------------------
def ssh_indi_getprop(ssh_user, ssh_host, ssh_port, indi_port, prop_str):
    try:
        ssh_command = [
            "ssh", "-p", str(ssh_port), f"{ssh_user}@{ssh_host}",
            "indi_getprop", "-p", str(indi_port), prop_str
        ]
        output = subprocess.check_output(ssh_command, text=True).strip()
        if "=" in output:
            return output.split("=")[-1].strip()
        else:
            return None
    except subprocess.CalledProcessError:
        return None

# -------------------- MAIN LOOP --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-user", default="stellarmate")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--indi-port", type=int, default=7624)
    parser.add_argument("--device", required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--to-redis", action="store_true")
    parser.add_argument("--udp-host")
    parser.add_argument("--udp-port", type=int)
    args = parser.parse_args()

    device = args.device

    if args.to_redis:
        r = redis.Redis()

    print(f"[capture_mount_ssh] Polling '{device}' via SSH {args.ssh_user}@{args.ssh_host}:{args.ssh_port} (indi_port={args.indi_port}) every {args.interval:.1f}s")

    while True:
        ra = ssh_indi_getprop(args.ssh_user, args.ssh_host, args.ssh_port, args.indi_port, f'"{device}.EQUATORIAL_EOD_COORD.RA"')
        dec = ssh_indi_getprop(args.ssh_user, args.ssh_host, args.ssh_port, args.indi_port, f'"{device}.EQUATORIAL_EOD_COORD.DEC"')

        if not ra or not dec:
            print("[capture_mount_ssh] Skipping due to missing RA/DEC")
            time.sleep(args.interval)
            continue

        try:
            ra_f = float(ra)
            dec_f = float(dec)
        except ValueError:
            print("[capture_mount_ssh] Invalid RA/DEC values")
            time.sleep(args.interval)
            continue

        now = Time.now()
        sc = SkyCoord(ra=ra_f*u.hourangle, dec=dec_f*u.deg, frame='icrs')
        altaz = sc.transform_to(AltAz(obstime=now, location=SITE))

        alt = altaz.alt.deg
        az = altaz.az.deg

        # Additional properties
        pier_west = ssh_indi_getprop(args.ssh_user, args.ssh_host, args.ssh_port, args.indi_port, f'"{device}.TELESCOPE_PIER_SIDE.PIER_WEST"')
        pier_east = ssh_indi_getprop(args.ssh_user, args.ssh_host, args.ssh_port, args.indi_port, f'"{device}.TELESCOPE_PIER_SIDE.PIER_EAST"')
        side = None
        if pier_west == "On":
            side = "WEST"
        elif pier_east == "On":
            side = "EAST"

        tracking = ssh_indi_getprop(args.ssh_user, args.ssh_host, args.ssh_port, args.indi_port, f'"{device}.TELESCOPE_TRACK_STATE.TRACK_ON"') == "On"

        iso = now.isot
        print(f"{iso} | RA: {ra_f:.6f} h | Dec: {dec_f:.6f}° | Alt: {alt:.2f}° | Az: {az:.2f}° | Side: {side} | Tracking: {tracking}")

        message_dict = {
            "timestamp": iso,
            "Computer_UTC": time.time(),
            "ra_hours": ra_f,
            "dec_deg": dec_f,
            "alt_deg": round(alt, 3),
            "az_deg": round(az, 3),
            "side": side,
            "tracking": str(tracking),
        }

        clean_dict = {k: v for k, v in message_dict.items() if v is not None}
        if len(clean_dict) < len(message_dict):
            skipped_keys = [k for k in message_dict if message_dict[k] is None]
            print(f"[Redis] Skipped keys with None values: {skipped_keys}")

        if args.to_redis:
            r.hset("MOUNT_GATTINI", mapping=clean_dict)

        if args.udp_host and args.udp_port:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            msg = str(clean_dict).encode()
            sock.sendto(msg, (args.udp_host, args.udp_port))

        time.sleep(args.interval)

if __name__ == "__main__":
    main()
