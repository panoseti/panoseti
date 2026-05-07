#!/bin/bash
# entrypoint-headnode-hw.sh — HITL headnode: SSH key setup as root, then
# delegate to entrypoint.sh for optional UID rewrite and exec gosu.
set -e

TARGET_UID="${LOCAL_UID:-1000}"
TARGET_GID="${LOCAL_GID:-1000}"

# Copy SSH keys from the bind-mounted host .ssh dir into the panoseti home dir.
mkdir -p /home/panoseti/.ssh
cp -rf /home/panoseti/.ssh-host/* /home/panoseti/.ssh/ 2>/dev/null || true
chown -R "${TARGET_UID}:${TARGET_GID}" /home/panoseti/.ssh
chmod 700 /home/panoseti/.ssh
find /home/panoseti/.ssh -type f -exec chmod 600 {} + 2>/dev/null || true

# Also copy to /root/.ssh so root-level ssh commands work (e.g. for gosu execs).
mkdir -p /root/.ssh
cp -rf /home/panoseti/.ssh-host/* /root/.ssh/ 2>/dev/null || true
chmod 700 /root/.ssh
find /root/.ssh -type f -exec chmod 600 {} + 2>/dev/null || true

# Ensure DAQ data directory exists and is owned by panoseti.
DATA_DIR="${DAQ_DATA_DIR:-/mnt/panoseti-test/}"
mkdir -p "${DATA_DIR}"
chown "${TARGET_UID}:${TARGET_GID}" "${DATA_DIR}"

# Ensure any metadata files created by root (like sw_info.json or flashuid) are fixed
[ -f /app/sw_info.json ] && chown "${TARGET_UID}:${TARGET_GID}" /app/sw_info.json
[ -f /app/flashuid ] && chown "${TARGET_UID}:${TARGET_GID}" /app/flashuid

exec /entrypoint.sh "$@"
