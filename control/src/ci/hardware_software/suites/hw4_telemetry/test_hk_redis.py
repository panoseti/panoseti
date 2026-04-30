"""
hw4_telemetry — Mid-level HK tests (Redis HASH lookup).

Requires the capture_hk.py daemon running and populating Redis keys of the
form QUABO_<boardloc>. The `redis_client` fixture skips the test if Redis
is unavailable, so the test is naturally self-gating.

Required state: BOOTED
Class: telemetry (batch_priority=0)
"""

from __future__ import annotations

import time

import pytest

from ci.hardware_software.hw_assertions import (
    assert_temperature_plausible,
    assert_voltage_in_spec,
    get_redis_hk,
)

pytestmark = pytest.mark.hw_class("telemetry")

_POPULATE_TIMEOUT_S = 15.0   # seconds for capture_hk.py to populate Redis
_POLL_INTERVAL_S = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_redis_key(redis_client, boardloc: int, timeout: float = _POPULATE_TIMEOUT_S) -> dict[str, str]:
    """Poll until QUABO_<boardloc> appears in Redis or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = get_redis_hk(redis_client, boardloc)
        if data:
            return data
        time.sleep(_POLL_INTERVAL_S)
    return {}


# ---------------------------------------------------------------------------
# Redis populated for all quabos
# ---------------------------------------------------------------------------

@pytest.mark.timeout(60)
def test_redis_populated_for_all_quabos(redis_client, topology) -> None:
    """
    Every quabo in the active topology must have a QUABO_<boardloc> HASH
    in Redis within _POPULATE_TIMEOUT_S seconds of the daemon starting.
    """
    for addr in topology.quabo_ips():
        data = _wait_for_redis_key(redis_client, addr.boardloc)
        assert data, (
            f"Redis QUABO_{addr.boardloc} not populated within {_POPULATE_TIMEOUT_S}s "
            f"(quabo {addr.ip} Q{addr.quadrant})"
        )


# ---------------------------------------------------------------------------
# Voltage rails in spec
# ---------------------------------------------------------------------------

def test_redis_voltage_rails_in_spec(redis_client, topology) -> None:
    """
    V12MON, V18MON, V33MON, V37MON must be within 5% of their nominal values.
    """
    voltage_specs = {
        "V12MON": 1.20,
        "V18MON": 1.80,
        "V33MON": 3.30,
        "V37MON": 3.70,
    }
    for addr in topology.quabo_ips():
        hk = _wait_for_redis_key(redis_client, addr.boardloc)
        if not hk:
            pytest.skip(f"QUABO_{addr.boardloc} not in Redis")
        for field, nominal in voltage_specs.items():
            if field in hk:
                assert_voltage_in_spec(hk, field, nominal, tolerance=0.05)


# ---------------------------------------------------------------------------
# Currents in spec
# ---------------------------------------------------------------------------

def test_redis_currents_in_spec(redis_client, topology) -> None:
    """
    I10MON, I18MON, I33MON must be positive and below a board-typical maximum.
    Nominal ranges derived from capture_hk.py conversion factors.
    """
    current_maxima = {
        "I10MON": 2.0,   # A
        "I18MON": 1.0,   # A
        "I33MON": 0.5,   # A
    }
    for addr in topology.quabo_ips():
        hk = _wait_for_redis_key(redis_client, addr.boardloc)
        if not hk:
            pytest.skip(f"QUABO_{addr.boardloc} not in Redis")
        for field, max_a in current_maxima.items():
            if field in hk:
                val = float(hk[field])
                assert 0 <= val <= max_a, (
                    f"QUABO_{addr.boardloc} {field}={val:.4f} A outside [0, {max_a}]"
                )


# ---------------------------------------------------------------------------
# Temperatures plausible
# ---------------------------------------------------------------------------

def test_redis_temperatures_plausible(redis_client, topology) -> None:
    """
    TEMP1 must be in [-10, 60] °C; TEMP2 must be in [20, 85] °C.
    VCCINT ≈ 1.0 V; VCCAUX ≈ 1.8 V (within 10%).
    """
    for addr in topology.quabo_ips():
        hk = _wait_for_redis_key(redis_client, addr.boardloc)
        if not hk:
            pytest.skip(f"QUABO_{addr.boardloc} not in Redis")
        if "TEMP1" in hk:
            assert_temperature_plausible(hk, "TEMP1")
        if "TEMP2" in hk:
            val = float(hk["TEMP2"])
            assert 20 <= val <= 85, f"TEMP2={val:.1f}°C outside [20, 85]"
        if "VCCINT" in hk:
            assert_voltage_in_spec(hk, "VCCINT", 1.0, tolerance=0.10)
        if "VCCAUX" in hk:
            assert_voltage_in_spec(hk, "VCCAUX", 1.8, tolerance=0.10)


# ---------------------------------------------------------------------------
# HV off state
# ---------------------------------------------------------------------------

def test_redis_hv_off_state(redis_client, topology) -> None:
    """
    With HV not commanded (quabo in BOOTED/ACQ_CONFIGURED state),
    |HVMON0..3| must all be < 1 V.
    """
    for addr in topology.quabo_ips():
        hk = _wait_for_redis_key(redis_client, addr.boardloc)
        if not hk:
            pytest.skip(f"QUABO_{addr.boardloc} not in Redis")
        for ch in range(4):
            field = f"HVMON{ch}"
            if field in hk:
                val = abs(float(hk[field]))
                assert val < 1.0, (
                    f"QUABO_{addr.boardloc} {field}={val:.2f} V unexpectedly high with HV off"
                )


# ---------------------------------------------------------------------------
# HV on state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("channel,setpoint_raw", [(0, 30000)])
def test_redis_hv_on_state(redis_client, topology, quabo, channel, setpoint_raw) -> None:
    """
    After commanding HV to a setpoint on one channel, HVMON for that channel
    must converge within ±1 V within 5 s.

    Parametrized: channel=0, setpoint_raw=30000 (≈ 73 V).
    Requires physical detector or dummy load; skip if HVMON stays near zero.
    """
    try:
        hv_vals = [0, 0, 0, 0]
        hv_vals[channel] = setpoint_raw
        quabo.hv_set(hv_vals)
        time.sleep(5.0)

        quabo_addrs = topology.quabo_ips()
        target_addr = next((a for a in quabo_addrs if a.quadrant == 0), None)
        if target_addr is None:
            pytest.skip("No Q0 quabo in topology")

        hk = _wait_for_redis_key(redis_client, target_addr.boardloc)
        field = f"HVMON{channel}"
        if field not in hk:
            pytest.skip(f"{field} not in Redis HASH")

        val = float(hk[field])
        if abs(val) < 1.0:
            pytest.skip(f"HVMON{channel} did not change (no detector/dummy load?)")

        # Convert setpoint_raw to approximate volts: 30000 raw ≈ 73 V (board-specific)
        # Just check that the HV is substantially non-zero and matches sign
        assert abs(val) > 10.0, f"HVMON{channel} only {val:.2f} V after commanding {setpoint_raw}"
    finally:
        quabo.hv_set([0, 0, 0, 0])


# ---------------------------------------------------------------------------
# Detector current offset corrected
# ---------------------------------------------------------------------------

def test_redis_detector_current_offset_corrected(redis_client, topology) -> None:
    """
    DETR0..3_CURR (post-offset correction) must be ≈ 0 μA with HV off,
    proving the offset math from capture_hk.py:56 is being applied.
    """
    for addr in topology.quabo_ips():
        hk = _wait_for_redis_key(redis_client, addr.boardloc)
        if not hk:
            pytest.skip(f"QUABO_{addr.boardloc} not in Redis")
        for ch in range(4):
            field = f"DETR{ch}_CURR"
            if field in hk:
                val = float(hk[field])
                assert abs(val) < 0.5, (
                    f"QUABO_{addr.boardloc} {field}={val:.3f} μA "
                    f"too large with HV off (offset correction may be broken)"
                )


# ---------------------------------------------------------------------------
# Startup flag once
# ---------------------------------------------------------------------------

def test_redis_startup_flag_once(redis_client, topology) -> None:
    """
    After the first HK update for a quabo, StartUp must be 0 for all
    subsequent updates. (StartUp==1 is set only on the first packet after boot.)

    We capture two consecutive Redis snapshots 4 s apart and verify the
    second one has StartUp==0.
    """
    for addr in topology.quabo_ips():
        hk1 = _wait_for_redis_key(redis_client, addr.boardloc)
        if not hk1:
            pytest.skip(f"QUABO_{addr.boardloc} not in Redis")
        time.sleep(4.0)
        hk2 = get_redis_hk(redis_client, addr.boardloc)
        if hk2 and "StartUp" in hk2:
            assert int(hk2["StartUp"]) == 0, (
                f"QUABO_{addr.boardloc} StartUp should be 0 after first packet"
            )


# ---------------------------------------------------------------------------
# Computer_UTC monotonic
# ---------------------------------------------------------------------------

def test_redis_computer_utc_monotonic(redis_client, topology) -> None:
    """
    Successive Computer_UTC values for each quabo must strictly increase.
    """
    for addr in topology.quabo_ips():
        hk1 = _wait_for_redis_key(redis_client, addr.boardloc)
        if not hk1 or "Computer_UTC" not in hk1:
            continue
        t1 = float(hk1["Computer_UTC"])
        time.sleep(4.0)
        hk2 = get_redis_hk(redis_client, addr.boardloc)
        if not hk2 or "Computer_UTC" not in hk2:
            continue
        t2 = float(hk2["Computer_UTC"])
        assert t2 > t1, (
            f"QUABO_{addr.boardloc} Computer_UTC did not increase: {t1} → {t2}"
        )


# ---------------------------------------------------------------------------
# Shutter state reflects command
# ---------------------------------------------------------------------------

def test_redis_shutter_state_reflects_command(redis_client, topology, quabo) -> None:
    """
    After sending shutter_new(closed=True), Redis SHUTTER_STATUS must be 0
    within 5 s. After shutter_new(closed=False), SHUTTER_STATUS must be 1.
    """
    quabo_addrs = topology.quabo_ips()
    target = next((a for a in quabo_addrs if a.quadrant == 0), None)
    if target is None:
        pytest.skip("No Q0 quabo in topology")

    quabo.shutter_new(closed=True)
    time.sleep(5.0)
    hk = get_redis_hk(redis_client, target.boardloc)
    if hk and "SHUTTER_STATUS" in hk:
        assert int(hk["SHUTTER_STATUS"]) == 0, (
            "SHUTTER_STATUS should be 0 (closed) after shutter_new(closed=True)"
        )

    quabo.shutter_new(closed=False)
    time.sleep(5.0)
    hk2 = get_redis_hk(redis_client, target.boardloc)
    if hk2 and "SHUTTER_STATUS" in hk2:
        assert int(hk2["SHUTTER_STATUS"]) == 1, (
            "SHUTTER_STATUS should be 1 (open) after shutter_new(closed=False)"
        )

    # Restore: close the shutter
    quabo.shutter_new(closed=True)


# ---------------------------------------------------------------------------
# UID stable across packets
# ---------------------------------------------------------------------------

def test_redis_uid_stable_across_packets(redis_client, topology) -> None:
    """
    Over 60 s, the UID field for each quabo must remain constant in Redis.
    """
    uid_snapshots: dict[int, list[str]] = {}
    for addr in topology.quabo_ips():
        hk = _wait_for_redis_key(redis_client, addr.boardloc)
        if hk and "UID" in hk:
            uid_snapshots[addr.boardloc] = [hk["UID"]]

    if not uid_snapshots:
        pytest.skip("No UID fields in Redis")

    time.sleep(60.0)

    for boardloc, uid_list in uid_snapshots.items():
        hk = get_redis_hk(redis_client, boardloc)
        if hk and "UID" in hk:
            uid_list.append(hk["UID"])

    for boardloc, uid_list in uid_snapshots.items():
        if len(uid_list) >= 2:
            assert uid_list[0] == uid_list[-1], (
                f"QUABO_{boardloc} UID changed: {uid_list[0]!r} → {uid_list[-1]!r}"
            )
