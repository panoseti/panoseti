#!/bin/bash
# entrypoint.sh — Clean, user-native entrypoint.
#
# No more runtime UID remapping. The image is now built with the correct
# LOCAL_UID/LOCAL_GID via build-time injection.
set -e

# Docker socket permission fix: match 'docker' group GID to host socket GID
# This is the only runtime 'root' task remaining, and only if the socket is mounted.
if [ "$(id -u)" = "0" ]; then
    if [ -S /var/run/docker.sock ]; then
        DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)
        groupmod -o -g "$DOCKER_GID" docker 2>/dev/null || true
        usermod -aG docker panoseti 2>/dev/null || true
    fi
    exec gosu panoseti "$@"
else
    # Correctly booting as panoseti (standard build-time UID injection path)
    exec "$@"
fi
