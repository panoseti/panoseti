#!/bin/bash
# entrypoint.sh — Clean, user-native entrypoint.
#
# No more runtime UID remapping. The image is now built with the correct
# LOCAL_UID/LOCAL_GID via build-time injection.
set -e

# Docker socket permission fix: match 'docker' group GID to host socket GID
# This is the only runtime 'root' task remaining, and only if the socket is mounted.
if [ "$(id -u)" = "0" ]; then
    if [ -n "$LOCAL_UID" ] && [ -n "$LOCAL_GID" ]; then
        groupmod -o -g "$LOCAL_GID" panoseti 2>/dev/null || true
        usermod -o -u "$LOCAL_UID" panoseti 2>/dev/null || true
    fi

    if [ -S /var/run/docker.sock ]; then
        DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)
        groupmod -o -g "$DOCKER_GID" docker 2>/dev/null || true
        usermod -aG docker panoseti 2>/dev/null || true
    fi

    # Always align ownership of application-critical directories to the
    # (potentially remapped) panoseti user.
    echo "Syncing ownership of /app /grpc /pypff /opt/venv /tmp..."
    chown -R panoseti:panoseti /app /grpc /pypff /opt/venv /tmp 2>/dev/null || true

    # get_logger() writes {service}.jsonl under here (tailed by Alloy). It's a
    # bind mount from the host, so Docker auto-creates it as root:root the
    # first time -- without this, every service silently falls back to
    # /tmp/panoseti_logs (container-local, invisible to Alloy) instead of
    # erroring, which is easy to miss.
    if [ -d /var/log/panoseti ]; then
        chown -R panoseti:panoseti /var/log/panoseti 2>/dev/null || true
    fi

    # Recursively claim the entire data mount point if it exists
    # This aligns files created by rsync (root) to the runtime user.
    DATA_DIR="${DAQ_DATA_DIR:-/mnt/panoseti-test}"
    if [ -d "$DATA_DIR" ]; then
        echo "Syncing ownership of $DATA_DIR..."
        chown -R panoseti:panoseti "$DATA_DIR" 2>/dev/null || true
    fi

    exec gosu panoseti "$@"
else
    # Correctly booting as panoseti (standard build-time UID injection path)
    exec "$@"
fi
