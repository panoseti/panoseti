#!/bin/bash
# entrypoint-daqnode.sh — DAQ-node startup.
set -e

DATA_DIR="${DAQ_DATA_DIR:-/data}"
mkdir -p "${DATA_DIR}"
cp /usr/local/lib/panoseti_hashpipe.so "${DATA_DIR}/hashpipe.so"

# Remove stale config files to ensure StartDaq can recreate them.
rm -f "${DATA_DIR}/module.config"

exec /entrypoint.sh "$@"
