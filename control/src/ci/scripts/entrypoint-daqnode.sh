#!/bin/bash
# entrypoint-daqnode.sh — DAQ-node startup: copy hashpipe.so to data dir, fix
# ownership, then delegate to the common entrypoint for optional UID rewrite.
set -e

DATA_DIR="${DAQ_DATA_DIR:-/data}"
mkdir -p "${DATA_DIR}"
cp /usr/local/lib/panoseti_hashpipe.so "${DATA_DIR}/hashpipe.so"
chown -R panoseti:panoseti "${DATA_DIR}"

exec /entrypoint.sh "$@"
