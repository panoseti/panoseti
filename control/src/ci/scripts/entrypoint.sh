#!/bin/bash
# entrypoint.sh — runtime UID/GID rewrite for dev mode, then exec gosu.
#
# In CI the image is used as-is (panoseti UID=1000); LOCAL_UID is unset → no-op.
# In dev mode the compose .dev.yml overlay sets LOCAL_UID=$(id -u) / LOCAL_GID=$(id -g)
# so files written to bind-mounted /app or /grpc are owned by the host user.
set -e

BAKED_UID=$(id -u panoseti 2>/dev/null || echo 1000)
BAKED_GID=$(id -g panoseti 2>/dev/null || echo 1000)

if [ -n "${LOCAL_GID}" ] && [ "${LOCAL_GID}" != "${BAKED_GID}" ]; then
    groupmod -o -g "${LOCAL_GID}" panoseti 2>/dev/null || true
fi

if [ -n "${LOCAL_UID}" ] && [ "${LOCAL_UID}" != "${BAKED_UID}" ]; then
    usermod -o -u "${LOCAL_UID}" panoseti 2>/dev/null || true
    # Fix ownership of internal directories that must be writable.
    # We do this as root before switching to the panoseti user via gosu.
    chown panoseti:panoseti /app /app/state /app/tmp /app/.pytest_cache /grpc /opt/venv 2>/dev/null || true
    # Recursive chown for state and tmp as they are expected to be small.
    chown -R panoseti:panoseti /app/state /app/tmp /app/.pytest_cache 2>/dev/null || true
fi

exec gosu panoseti "$@"
