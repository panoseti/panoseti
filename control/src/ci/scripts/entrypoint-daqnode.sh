#!/bin/bash
# entrypoint-daqnode.sh — DAQ-node startup.
set -e

DATA_DIR="${DAQ_DATA_DIR:-/data}"
mkdir -p "${DATA_DIR}"
cp /usr/local/lib/panoseti_hashpipe.so "${DATA_DIR}/hashpipe.so"

# Remove stale config files to ensure StartDaq can recreate them.
rm -f "${DATA_DIR}/module.config"

# Align ownership if running as root
if [ "$(id -u)" = "0" ]; then
    if [ -n "$LOCAL_UID" ] && [ -n "$LOCAL_GID" ]; then
        groupmod -o -g "$LOCAL_GID" panoseti 2>/dev/null || true
        usermod -o -u "$LOCAL_UID" panoseti 2>/dev/null || true
    fi
    chown -R panoseti:panoseti "${DATA_DIR}" 2>/dev/null || true
fi

exec /entrypoint.sh "$@"
