#!/bin/sh
# gateway_setup.sh — socat TCP forwarding rules for the integration test gateway.
#
# The gateway container bridges headnode_net (10.0.1.0/24) and daqnode_net (192.168.0.0/24).
# Traffic arriving at the gateway on port 50051 is forwarded to the daqnode at 192.168.0.10.
#
# Hardware CI: replace daqnode address with real DAQ node IP.

set -e

DAQNODE="${DAQNODE_IP:-192.168.0.10}"

echo "Starting socat forwarders to daqnode at ${DAQNODE}"

# gRPC: daq_control service
socat TCP-LISTEN:50051,fork,reuseaddr TCP:"${DAQNODE}":50051 &
PID1=$!

# gRPC: daq_data service
socat TCP-LISTEN:50052,fork,reuseaddr TCP:"${DAQNODE}":50052 &
PID2=$!

echo "Forwarding gRPC ports 50051 (daq_control) and 50052 (daq_data) to ${DAQNODE}"

# Wait for any child to exit
wait $PID1 $PID2
