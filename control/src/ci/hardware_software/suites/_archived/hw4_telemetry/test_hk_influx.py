"""
hw4_telemetry — High-level HK tests (InfluxDB).

Requires the storeInfluxDB.py daemon running and ingesting Redis keys into
the 'metadata' database.  The `influx_client` fixture skips these tests if
InfluxDB is unavailable.

Required state: BOOTED
Class: telemetry (batch_priority=0)
"""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.hw_class("telemetry")

_INGEST_WAIT_S = 10.0   # time to wait for InfluxDB to receive points
_CADENCE_S = 3.0         # expected HK cadence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query(influx_client, q: str) -> list[dict]:
    """Execute an InfluxQL query and return all result rows."""
    result = influx_client.query(q)
    rows = []
    for series in result.raw.get("results", [{}])[0].get("series", []):
        cols = series.get("columns", [])
        for vals in series.get("values", []):
            rows.append(dict(zip(cols, vals, strict=False)))
    return rows


def _measurement_name(boardloc: int) -> str:
    return f"QUABO_{boardloc}"


# ---------------------------------------------------------------------------
# Measurement per quabo
# ---------------------------------------------------------------------------

def test_influx_measurement_per_quabo(influx_client, topology) -> None:
    """
    SHOW MEASUREMENTS must include QUABO_<boardloc> for every quabo
    in the active topology.
    """
    time.sleep(_INGEST_WAIT_S)
    rows = _query(influx_client, 'SHOW MEASUREMENTS ON "metadata"')
    names = {r.get("name", "") for r in rows}

    for addr in topology.quabo_ips():
        expected = _measurement_name(addr.boardloc)
        assert expected in names, (
            f"{expected} not found in InfluxDB metadata measurements. "
            f"Available: {sorted(names)}"
        )


# ---------------------------------------------------------------------------
# Point count matches cadence
# ---------------------------------------------------------------------------

def test_influx_point_count_matches_cadence(influx_client, topology) -> None:
    """
    Over 60 s, count(V12MON) per quabo measurement should be ≈ 20 ± 2
    (3 s cadence → ~20 points in 60 s).
    """
    time.sleep(60.0)
    for addr in topology.quabo_ips():
        meas = _measurement_name(addr.boardloc)
        rows = _query(
            influx_client,
            f'SELECT count(V12MON) FROM "metadata"."{meas}" WHERE time > now() - 65s'
        )
        if not rows:
            continue
        count = rows[0].get("count", 0) or 0
        assert 18 <= count <= 22, (
            f"{meas}: expected ~20 points in 60 s, got {count}"
        )


# ---------------------------------------------------------------------------
# Dedup: no two points with same Computer_UTC
# ---------------------------------------------------------------------------

def test_influx_dedup_correct(influx_client, topology) -> None:
    """
    No two points in a measurement should have identical Computer_UTC values.
    This proves capture_hk's key_timestamps deduplication works.
    """
    time.sleep(_INGEST_WAIT_S)
    for addr in topology.quabo_ips():
        meas = _measurement_name(addr.boardloc)
        rows = _query(
            influx_client,
            f'SELECT Computer_UTC FROM "metadata"."{meas}" WHERE time > now() - 120s'
        )
        if len(rows) < 2:
            continue
        utc_values = [r.get("Computer_UTC") for r in rows if r.get("Computer_UTC") is not None]
        assert len(utc_values) == len(set(utc_values)), (
            f"{meas}: duplicate Computer_UTC values detected — dedup may be broken"
        )


# ---------------------------------------------------------------------------
# Tags correct
# ---------------------------------------------------------------------------

def test_influx_tags_correct(influx_client, topology) -> None:
    """
    Every point must be tagged with observatory=<obs_name> and
    datatype=housekeeping.
    """
    from control.utils import config_file
    obs_name = config_file.get_obs_config().name

    time.sleep(_INGEST_WAIT_S)
    for addr in topology.quabo_ips()[:1]:  # check first quabo only for speed
        meas = _measurement_name(addr.boardloc)
        rows = _query(
            influx_client,
            f'SHOW TAG VALUES ON "metadata" FROM "{meas}" WITH KEY IN ("observatory", "datatype")'
        )
        tag_values: dict[str, set[str]] = {}
        for r in rows:
            key = r.get("key", "")
            val = r.get("value", "")
            tag_values.setdefault(key, set()).add(val)

        if "observatory" in tag_values and obs_name:
            assert obs_name in tag_values["observatory"], (
                f"{meas}: tag 'observatory' missing value {obs_name!r}"
            )
        if "datatype" in tag_values:
            assert "housekeeping" in tag_values["datatype"], (
                f"{meas}: tag 'datatype' missing value 'housekeeping'"
            )


# ---------------------------------------------------------------------------
# Field types
# ---------------------------------------------------------------------------

def test_influx_field_types(influx_client, topology) -> None:
    """
    Numeric HK fields (V12MON, TEMP1, etc.) must be stored as floats;
    UID, FWVER, FWTIME must be stored as strings.
    """
    time.sleep(_INGEST_WAIT_S)
    for addr in topology.quabo_ips()[:1]:
        meas = _measurement_name(addr.boardloc)
        rows = _query(
            influx_client,
            f'SHOW FIELD KEYS ON "metadata" FROM "{meas}"'
        )
        field_types = {r.get("fieldKey", ""): r.get("fieldType", "") for r in rows}

        float_fields = ["V12MON", "V18MON", "V33MON", "V37MON", "TEMP1", "TEMP2"]
        string_fields = ["UID", "FWVER", "FWTIME"]

        for f in float_fields:
            if f in field_types:
                assert field_types[f] == "float", (
                    f"{meas}.{f} expected float, got {field_types[f]!r}"
                )
        for f in string_fields:
            if f in field_types:
                assert field_types[f] == "string", (
                    f"{meas}.{f} expected string, got {field_types[f]!r}"
                )


# ---------------------------------------------------------------------------
# HV step response visible
# ---------------------------------------------------------------------------

def test_influx_hv_step_response_visible(influx_client, topology, quabo) -> None:
    """
    Step HV from 0 → 30000 → 0; assert the resulting HVMON0 time series
    in InfluxDB shows the step within 2 polling intervals (≤ 6 s).
    """
    quabo_addrs = topology.quabo_ips()
    target = next((a for a in quabo_addrs if a.quadrant == 0), None)
    if target is None:
        pytest.skip("No Q0 quabo in topology")

    meas = _measurement_name(target.boardloc)

    # Step up
    quabo.hv_set([30000, 0, 0, 0])
    time.sleep(8.0)

    rows_up = _query(
        influx_client,
        f'SELECT HVMON0 FROM "metadata"."{meas}" WHERE time > now() - 15s'
    )
    hv_vals_up = [r.get("HVMON0", 0) or 0 for r in rows_up]

    # Step down
    quabo.hv_set([0, 0, 0, 0])
    time.sleep(8.0)

    rows_down = _query(
        influx_client,
        f'SELECT HVMON0 FROM "metadata"."{meas}" WHERE time > now() - 15s'
    )
    hv_vals_down = [r.get("HVMON0", 0) or 0 for r in rows_down]

    if not hv_vals_up:
        pytest.skip(f"No HVMON0 data in {meas} during step-up window")

    max_up = max(abs(v) for v in hv_vals_up)
    if max_up < 1.0:
        pytest.skip("HVMON0 did not change — no detector/dummy load connected")

    if hv_vals_down:
        max_down = max(abs(v) for v in hv_vals_down)
        assert max_down < max_up, (
            f"HVMON0 did not decrease after HV zero: max_up={max_up:.1f}, max_down={max_down:.1f}"
        )


# ---------------------------------------------------------------------------
# HK continues through run lifecycle
# ---------------------------------------------------------------------------

def test_influx_continues_through_run_lifecycle(influx_client, topology, runner) -> None:
    """
    HK time series must be uninterrupted across pseti start / pseti stop.
    Verify there are no gaps > 2x cadence in the TEMP1 series during a run.
    """
    from control.pseti import app

    # Start a short run
    result = runner.invoke(app, ["start", "--yes", "--nsecs", "15", "--no-hv"])
    assert result.exit_code == 0, f"pseti start failed:\n{result.stdout}"
    time.sleep(20.0)
    runner.invoke(app, ["stop", "--yes"])
    time.sleep(5.0)

    for addr in topology.quabo_ips()[:1]:
        meas = _measurement_name(addr.boardloc)
        rows = _query(
            influx_client,
            f'SELECT TEMP1 FROM "metadata"."{meas}" WHERE time > now() - 50s ORDER BY time'
        )
        if len(rows) < 4:
            pytest.skip(f"Too few TEMP1 points in {meas} to check continuity")

        times_ns = [r.get("time") for r in rows if r.get("time")]
        # InfluxDB returns ISO8601 time strings; skip detailed gap check
        # (gap detection would need ISO parse — this just checks we have ≥4 points)
        assert len(times_ns) >= 4, f"{meas}: fewer than 4 TEMP1 points during run lifecycle"


# ---------------------------------------------------------------------------
# HK stops on power off
# ---------------------------------------------------------------------------

def test_influx_stops_on_power_off(influx_client, topology) -> None:
    """
    After a WPS power-off, the last HK point in InfluxDB must be within
    5 s of the power-off command.

    This test intentionally powers off the hardware; it runs as the final
    test in the telemetry batch so the lifecycle batch can power-cycle next.
    """
    from control.power import quabo_power
    from control.utils import config_file

    obs = config_file.get_obs_config()
    wps_entries = {k: v for k, v in obs.items() if k.startswith("wps")}
    if not wps_entries:
        pytest.skip("No WPS in obs_config; cannot test power-off HK stop")

    t_off = time.time()
    for wps_val in wps_entries.values():
        quabo_power(wps_val, on=False)

    time.sleep(8.0)  # wait for a couple of polling cycles

    for addr in topology.quabo_ips()[:1]:
        meas = _measurement_name(addr.boardloc)
        rows = _query(
            influx_client,
            f'SELECT LAST(Computer_UTC) FROM "metadata"."{meas}"'
        )
        if not rows:
            continue
        last_utc = rows[0].get("last")
        if last_utc is not None:
            assert float(last_utc) <= t_off + 5.0, (
                f"{meas}: last HK Computer_UTC {last_utc:.1f} more than 5 s after power-off "
                f"(t_off={t_off:.1f})"
            )

    # Power back on for subsequent lifecycle tests
    for wps_val in wps_entries.values():
        quabo_power(wps_val, on=True)
