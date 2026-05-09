#!/bin/bash
# entrypoint-daqnode.sh — DAQ-node startup: copy hashpipe.so to data dir, fix
# ownership, then delegate to the common entrypoint for optional UID rewrite.
set -e

DATA_DIR="${DAQ_DATA_DIR:-/data}"
mkdir -p "${DATA_DIR}"
cp /usr/local/lib/panoseti_hashpipe.so "${DATA_DIR}/hashpipe.so"

# Remove stale config files that might be owned by a different UID
# to ensure StartDaq can recreate them.
rm -f "${DATA_DIR}/module.config"

# Only attempt chown if we are root
if [ "$(id -u)" = "0" ]; then
    TARGET_UID="${LOCAL_UID:-1000}"
    TARGET_GID="${LOCAL_GID:-1000}"
    chown "${TARGET_UID}:${TARGET_GID}" "${DATA_DIR}"
    chown "${TARGET_UID}:${TARGET_GID}" "${DATA_DIR}/hashpipe.so"
fi

exec /entrypoint.sh "$@"
