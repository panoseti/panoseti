#!/bin/bash
# entrypoint-daqnode.sh — DAQ-node startup: copy hashpipe.so to data dir, fix
# ownership, then delegate to the common entrypoint for optional UID rewrite.
set -e

DATA_DIR="${DAQ_DATA_DIR:-/data}"
mkdir -p "${DATA_DIR}"
cp /usr/local/lib/panoseti_hashpipe.so "${DATA_DIR}/hashpipe.so"

TARGET_UID="${LOCAL_UID:-1000}"
TARGET_GID="${LOCAL_GID:-1000}"

chown "${TARGET_UID}:${TARGET_GID}" "${DATA_DIR}"
chown "${TARGET_UID}:${TARGET_GID}" "${DATA_DIR}/hashpipe.so"

exec /entrypoint.sh "$@"
