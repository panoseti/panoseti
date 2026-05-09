#!/bin/bash
# entrypoint-headnode-hw.sh — HITL headnode: SSH key setup as root, then
# delegate to entrypoint.sh for optional UID rewrite and exec gosu.
set -e

# Only perform root-level setup (SSH keys, UID dance) if we are currently root.
if [ "$(id -u)" = "0" ]; then
    TARGET_UID="${LOCAL_UID:-1000}"
    TARGET_GID="${LOCAL_GID:-1000}"

    # Copy SSH keys from the bind-mounted host .ssh dir into the panoseti home dir.
    mkdir -p /home/panoseti/.ssh
    cp -rf /home/panoseti/.ssh-host/* /home/panoseti/.ssh/ 2>/dev/null || true
    chown -R "${TARGET_UID}:${TARGET_GID}" /home/panoseti/.ssh 2>/dev/null || true
    chmod 700 /home/panoseti/.ssh 2>/dev/null || true
    find /home/panoseti/.ssh -type f -exec chmod 600 {} + 2>/dev/null || true

    # Also copy to /root/.ssh so root-level ssh commands work (e.g. for gosu execs).
    mkdir -p /root/.ssh
    cp -rf /home/panoseti/.ssh-host/* /root/.ssh/ 2>/dev/null || true
    chmod 700 /root/.ssh 2>/dev/null || true
    find /root/.ssh -type f -exec chmod 600 {} + 2>/dev/null || true

    # Ensure DAQ data directory exists and is owned by panoseti.
    DATA_DIR="${DAQ_DATA_DIR:-/mnt/panoseti-test/}"
    mkdir -p "${DATA_DIR}"
    chown "${TARGET_UID}:${TARGET_GID}" "${DATA_DIR}" 2>/dev/null || true

    # Ensure any metadata files created by root (like sw_info.json or flashuid) are fixed
    [ -f /app/sw_info.json ] && chown "${TARGET_UID}:${TARGET_GID}" /app/sw_info.json 2>/dev/null || true
    [ -f /app/flashuid ] && chown "${TARGET_UID}:${TARGET_GID}" /app/flashuid 2>/dev/null || true
fi

exec /entrypoint.sh "$@"
