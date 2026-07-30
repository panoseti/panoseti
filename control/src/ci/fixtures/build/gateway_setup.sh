#!/bin/sh
# gateway_setup.sh — socat TCP forwarding rules for the integration test gateway.
#
# The gateway container bridges headnode_net (10.0.1.0/24) and daqnode_net (192.168.0.0/24).
# The unified gRPC server on the daqnode hosts daq_data + daq_control on a single port (50051),
# so only one forwarding rule is needed.
#
# Hardware CI: replace daqnode address with real DAQ node IP.

set -e

DAQNODE="${DAQNODE_IP:-192.168.0.10}"

echo "Starting socat forwarder to daqnode at ${DAQNODE}"

# gRPC: unified server — daq_data + daq_control on port 50051
socat TCP-LISTEN:50051,fork,reuseaddr TCP:"${DAQNODE}":50051

