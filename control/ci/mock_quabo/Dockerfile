FROM python:3.14-slim

# Install ip utilities for adding IP aliases at startup
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what the server needs
COPY control/ci/mock_quabo/server.py /app/server.py

# Environment defaults (override in docker-compose)
ENV MOCK_QUABO_BASE_IP=0.0.0.0
ENV MOCK_QUABO_MODULE_ID=200
ENV MOCK_QUABO_UDS=/tmp/mock_quabo.sock

# Startup: add IP aliases for all 4 quabo IPs in the module, then launch server.
# The base IP is the module base (quabo 0). Quabos 1-3 are base+1, base+2, base+3.
# Requires NET_ADMIN capability.
COPY control/ci/mock_quabo/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 60000/udp 60001/udp 60002/udp 60003/udp

CMD ["/app/entrypoint.sh"]
