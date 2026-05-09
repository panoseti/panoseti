#!/bin/bash
# entrypoint.sh — runtime UID/GID rewrite for dev mode, then exec gosu.
#
# In CI the image is used as-is (panoseti UID=1000); LOCAL_UID is unset → no-op.
# In dev mode the compose .dev.yml overlay sets LOCAL_UID=$(id -u) / LOCAL_GID=$(id -g)
# so files written to bind-mounted /app or /grpc are owned by the host user.
set -e

BAKED_UID=$(id -u panoseti 2>/dev/null || echo 1000)
BAKED_GID=$(id -g panoseti 2>/dev/null || echo 1000)

# Only perform UID/GID dance if we are currently root.
if [ "$(id -u)" = "0" ]; then
    if [ -n "${LOCAL_GID}" ] && [ "${LOCAL_GID}" != "${BAKED_GID}" ]; then
        groupmod -o -g "${LOCAL_GID}" panoseti 2>/dev/null || true
    fi

    if [ -n "${LOCAL_UID}" ] && [ "${LOCAL_UID}" != "${BAKED_UID}" ]; then
        usermod -o -u "${LOCAL_UID}" panoseti 2>/dev/null || true
        # Recursive chown of /app, /grpc, and /opt/venv ensures that files baked at UID 1000 
        # are aligned with the runtime LOCAL_UID.
        chown -R panoseti:panoseti /app /grpc /opt/venv 2>/dev/null || true
    fi

    # Docker socket permission fix: match 'docker' group GID to host socket GID
    if [ -S /var/run/docker.sock ]; then
        DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)
        groupmod -o -g "$DOCKER_GID" docker 2>/dev/null || true
        usermod -aG docker panoseti 2>/dev/null || true
    fi

    exec gosu panoseti "$@"
else
    # Already non-root, just exec.
    exec "$@"
fi
