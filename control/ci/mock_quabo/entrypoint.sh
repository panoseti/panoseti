#!/bin/sh
# Add IP aliases for quabos 1-3 within the module (quabo 0 = primary container IP)
# MOCK_QUABO_BASE_IP is set to the module base IP (last octet = multiple of 4).
# The container's actual IP is the same as MOCK_QUABO_BASE_IP.
# We add .+1, .+2, .+3 as aliases on eth0 (or the primary interface).

set -e

if [ -n "$MOCK_QUABO_BASE_IP" ] && [ "$MOCK_QUABO_BASE_IP" != "0.0.0.0" ]; then
    # Determine the primary interface
    IFACE=$(ip route | awk '/default/ {print $5; exit}')
    if [ -z "$IFACE" ]; then
        IFACE="eth0"
    fi

    # Parse base IP octets
    BASE_PREFIX=$(echo "$MOCK_QUABO_BASE_IP" | cut -d'.' -f1-3)
    BASE_LAST=$(echo "$MOCK_QUABO_BASE_IP" | cut -d'.' -f4)

    # Add aliases for quabos 1, 2, 3
    for OFFSET in 1 2 3; do
        ALIAS_IP="${BASE_PREFIX}.$((BASE_LAST + OFFSET))"
        ip addr add "${ALIAS_IP}/24" dev "$IFACE" 2>/dev/null || true
        echo "Added alias: ${ALIAS_IP}"
    done
fi

exec python /app/server.py
