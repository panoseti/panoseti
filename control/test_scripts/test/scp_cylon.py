#!/usr/bin/env python3
import os
import subprocess
from datetime import datetime

# === CONFIGURATION ===
local_base = "/mnt/data11/data/palomar/L0"
remote_host = "132.239.146.24"
remote_user = "panoseti"
remote_base = "/home/panoseti/www/DATA"
remote_base_ssh = "www/DATA"
bandwidth_limit_kbps = 40000   # limit (kbit/s)

# === DETECT DATE FOLDER (from local path name) ===
local_date_dir = "/mnt/data11/data/palomar/L0/20251014"  # <- or dynamically built
yyyymmdd = os.path.basename(local_date_dir)
remote_path = f"{remote_base}/{yyyymmdd}"
remote_path_ssh = f"{remote_base_ssh}/{yyyymmdd}"

# === SETUP LOGGING ===
log_dir = "/mnt/data11/data/palomar/L0/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"scp_{yyyymmdd}.log")

def log(msg):
    """Append message with timestamp to log file and print to console."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(log_file, "a") as f:
        f.write(line + "\n")

log(f"=== Starting SCP transfer for {yyyymmdd} ===")
log(f"Local path: {local_date_dir}")
log(f"Remote path: {remote_user}@{remote_host}:{remote_path}")
log(f"Bandwidth limit: {bandwidth_limit_kbps} kbit/s")
log(f"Log file: {log_file}")

# === CHECK LOCAL DIRECTORY ===
if not os.path.exists(local_date_dir):
    log(f"? ERROR: Local directory {local_date_dir} not found.")
    raise SystemExit(1)

# === CREATE REMOTE DIRECTORY ===
mkdir_cmd = f'ssh {remote_user}@{remote_host} "mkdir -p {remote_path_ssh}"'
log(f"Creating remote directory (if missing): {mkdir_cmd}")
mkdir_result = subprocess.run(mkdir_cmd, shell=True, capture_output=True, text=True)
if mkdir_result.returncode != 0:
    log(f"? ERROR creating remote directory:\n{mkdir_result.stderr}")
    raise SystemExit(1)
log("? Remote directory ready.")

# === PERFORM SCP TRANSFER ===
scp_cmd = (
    f"scp -l {bandwidth_limit_kbps} -r {local_date_dir}/* "
    f"{remote_user}@{remote_host}:{remote_path}/"
)
log(f"Executing: {scp_cmd}")
scp_result = subprocess.run(scp_cmd, shell=True, capture_output=True, text=True)

# === LOG OUTPUT ===
if scp_result.stdout:
    log("---- SCP STDOUT ----\n" + scp_result.stdout.strip())
if scp_result.stderr:
    log("---- SCP STDERR ----\n" + scp_result.stderr.strip())

# === STATUS ===
if scp_result.returncode == 0:
    log("? Transfer completed successfully.")
else:
    log(f"? Transfer failed with code {scp_result.returncode}.")

log("=== SCP session finished ===\n")
