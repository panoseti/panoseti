"""
hw_env_check — Pre-test environment validation (no hardware power required).

This suite runs before the boot sequence (batch_priority = -2) and validates
that the physical lab infrastructure is correctly wired and reachable.
A failure here means configuration or network issues must be fixed before
any hardware tests can succeed.

Checks:
  1. Config files valid — pseti val passes for all JSON configs.
  2. WPS reachable — HTTP GET to each WPS outlet returns a 200-class response.
  3. DAQ node reachable — TCP/ICMP ping to each DAQ node IP.
  4. DAQ node SSH — passwordless SSH succeeds (required for remote commands).

Note: pseti val network (which pings quabos) is NOT run here because quabos
are only reachable after the boot sequence powers them on and loads firmware.

Required state: UNPOWERED (no hardware power changes).
Leaves state: UNPOWERED.
"""

from __future__ import annotations

import logging
import subprocess

import pytest

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.hw_class("env_check"),
    pytest.mark.timeout(60),
]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_validation(runner) -> None:
    """All JSON config files pass pseti val (schema + cross-config rules)."""
    from control.pseti import app
    result = runner.invoke(app, ["val"])
    assert result.exit_code == 0, (
        f"pseti val failed — fix config errors before running hardware tests:\n{result.output}"
    )


# ---------------------------------------------------------------------------
# WPS reachability
# ---------------------------------------------------------------------------

def test_wps_reachable(topology) -> None:
    """Each configured WPS outlet responds to an HTTP HEAD request."""
    outlets = topology.wps_outlets()
    if not outlets:
        pytest.skip("No WPS outlets defined in obs_config")

    errors = []
    for wps in outlets:
        if not wps.url:
            errors.append(f"{wps.name}: no URL configured")
            continue
        r = subprocess.run(
            ["curl", "-s", "--connect-timeout", "5", "--max-time", "10",
             "--head", "--write-out", "%{http_code}", "--output", "/dev/null", wps.url],
            capture_output=True, text=True, timeout=15,
        )
        http_code = r.stdout.strip()
        if r.returncode != 0 or not http_code.startswith("2"):
            errors.append(
                f"{wps.name} ({wps.url}): curl returned {r.returncode}, "
                f"HTTP {http_code or 'no response'}"
            )
        else:
            logger.info("WPS %s (%s): HTTP %s", wps.name, wps.url, http_code)

    assert not errors, (
        f"WPS reachability FAILED for {len(errors)} outlet(s):\n" + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# DAQ node reachability
# ---------------------------------------------------------------------------

def test_daq_node_reachable(topology) -> None:
    """Each DAQ node IP responds to ICMP ping."""
    nodes = topology.daq_nodes()
    if not nodes:
        pytest.skip("No DAQ nodes defined in daq_config")

    errors = []
    for node in nodes:
        r = subprocess.run(
            ["ping", "-c1", "-W3", node.host],
            capture_output=True, timeout=10,
        )
        if r.returncode != 0:
            errors.append(f"DAQ node {node.host} (modules {node.module_ids}): no ICMP response")
        else:
            logger.info("DAQ node %s: pingable", node.host)

    assert not errors, (
        f"DAQ node reachability FAILED for {len(errors)} node(s):\n" + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# DAQ node SSH
# ---------------------------------------------------------------------------

def test_daq_node_ssh(topology) -> None:
    """Passwordless SSH into each DAQ node succeeds (required for gRPC server start)."""
    nodes = topology.daq_nodes()
    if not nodes:
        pytest.skip("No DAQ nodes defined in daq_config")

    errors = []
    for node in nodes:
        r = subprocess.run(
            [
                "ssh",
                "-o", "BatchMode=yes",          # fail immediately if key prompt needed
                "-o", "ConnectTimeout=5",
                "-o", "StrictHostKeyChecking=no",
                node.host,
                "echo ok",
            ],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0 or r.stdout.strip() != "ok":
            errors.append(
                f"DAQ node {node.host}: SSH failed (exit {r.returncode}): {r.stderr.strip()}"
            )
        else:
            logger.info("DAQ node %s: SSH ok", node.host)

    assert not errors, (
        f"DAQ node SSH FAILED for {len(errors)} node(s):\n" + "\n".join(errors)
        + "\n\nEnsure the head node's public key is in authorized_keys on each DAQ node."
    )
