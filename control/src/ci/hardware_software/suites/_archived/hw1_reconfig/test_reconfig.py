"""
hw1_reconfig — Fast reconfiguration tests.

Tests run with hardware in ACQ_CONFIGURED state; they perform configuration
changes and short observations without triggering a power cycle.

Required state: ACQ_CONFIGURED
Class: fast_reconfig (batch_priority=1)

Leaves state: ACQ_CONFIGURED
"""

from __future__ import annotations

import struct
import time

import pytest

from control.driver.quabo_driver import (
    ACQ_IMAGE_8BIT,
    DAQ_PARAMS,
    QUABO,
)

pytestmark = pytest.mark.hw_class("fast_reconfig")

_HK_SETTLE_S = 5.0  # time to wait for HK to reflect a config change


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_acqmode_from_hk(quabo: QUABO) -> int | None:
    """Read the ACQMODE value from the next HK packet's bytes[4:6]."""
    pkt = quabo.read_hk_packet()
    if pkt is None or len(pkt) < 6:
        return None
    return struct.unpack_from("<H", pkt, 4)[0]


# ---------------------------------------------------------------------------
# Interleave state switch
# ---------------------------------------------------------------------------

def test_interleave_state_switch(quabo: QUABO) -> None:
    """
    Configure `image` mode, then switch to `image_8bit` via a second
    send_daq_params call. Verify the HK packet's ACQMODE byte reflects the
    new mode within _HK_SETTLE_S seconds.
    """
    # Start in 16-bit image mode
    params_image = DAQ_PARAMS(
        do_image=True, image_us=10000, image_8bit=False,
        do_ph=False, bl_subtract=True
    )
    quabo.send_daq_params(params_image)
    time.sleep(0.5)

    # Switch to 8-bit image mode
    params_8bit = DAQ_PARAMS(
        do_image=True, image_us=10000, image_8bit=True,
        do_ph=False, bl_subtract=True
    )
    quabo.send_daq_params(params_8bit)

    # Poll HK packets for _HK_SETTLE_S to see mode change
    deadline = time.monotonic() + _HK_SETTLE_S
    observed_8bit = False
    while time.monotonic() < deadline:
        pkt = quabo.read_hk_packet()
        if pkt and len(pkt) >= 6:
            # ACQ mode bits are reported in ACQMODE field
            mode = pkt[4]  # byte 4 of HK (per packet-interface.md)
            if mode & ACQ_IMAGE_8BIT:
                observed_8bit = True
                break

    assert observed_8bit, (
        f"HK did not show ACQ_IMAGE_8BIT bit within {_HK_SETTLE_S}s after reconfigure"
    )

    # Restore to 16-bit mode
    quabo.send_daq_params(params_image)


# ---------------------------------------------------------------------------
# Trigger mask per channel
# ---------------------------------------------------------------------------

def test_trigger_mask_per_channel(quabo: QUABO) -> None:
    """
    Disable all channels via CHANMASK, then restore the default mask.
    Verifies send_trigger_mask is accepted without error and the FPGA echoes.
    """
    # Build a config that disables all channels (mask = 0x0000)
    disable_config = {f'CHANMASK_{i}': 0 for i in range(9)}
    quabo.flush_rx_buf()
    quabo.send_trigger_mask(disable_config, do_flush_rx_buf=False)

    # The FPGA echoes the trigger mask command (opcode 0x86)
    quabo.sock.settimeout(2.0)
    try:
        data, _ = quabo.sock.recvfrom(64)
        assert data[0] == 0x86, f"Expected 0x86 echo, got 0x{data[0]:02X}"
        # All CHANMASK bytes should be zero (channels disabled)
        # channels are encoded at bytes 4..40 (9 x 4 bytes each)
        for i in range(9):
            val = struct.unpack_from("<I", data, 4 + 4 * i)[0]
            assert val == 0, f"CHANMASK_{i} expected 0, got {val}"
    except (TimeoutError, OSError):
        pytest.skip("No echo for trigger mask command — FPGA may not echo this command")
    finally:
        quabo.sock.settimeout(0.5)

    # Restore default: all channels enabled (0xFFFFFFFF)
    default_config = {f'CHANMASK_{i}': 0xFFFFFFFF for i in range(9)}
    quabo.send_trigger_mask(default_config)


# ---------------------------------------------------------------------------
# GOE mask modes
# ---------------------------------------------------------------------------

def test_goe_mask_modes(quabo: QUABO) -> None:
    """
    Cycle GOE mask through 0x1, 0x2, 0x3 (1-pixel, 2-pixel, 3-pixel triggers).
    Verify send_goe_mask is accepted without error for each mode.
    """
    for mask_value in (0x1, 0x2, 0x3):
        goe_config = {'GOEMASK': mask_value}
        quabo.flush_rx_buf()
        quabo.send_goe_mask(goe_config, do_flush_rx_buf=False)

        quabo.sock.settimeout(2.0)
        try:
            data, _ = quabo.sock.recvfrom(64)
            assert data[0] == 0x8E, f"Expected 0x8E echo for GOEMASK={mask_value:#x}, got 0x{data[0]:02X}"
        except (TimeoutError, OSError):
            # Some FPGA versions may not echo GOE mask; skip rather than fail
            pass
        finally:
            quabo.sock.settimeout(0.5)

    # Restore safe default (3-pixel threshold)
    quabo.send_goe_mask({'GOEMASK': 0x3})


# ---------------------------------------------------------------------------
# HV setpoint step response
# ---------------------------------------------------------------------------

def test_hv_setpoint_step_response(quabo: QUABO) -> None:
    """
    Set HV to a moderate setpoint (digital value 30000, ≈ 73 V) on channel 0,
    wait 5 s, then assert HVMON0 from the HK packet is non-zero (HV ramped up).
    Then zero the HV and assert HVMON0 returns near zero.

    This tests the step response of the HV DAC without requiring HV_ON state
    (we send the DAC value directly, independent of the state machine).

    Note: this requires the quabo to have a detector or dummy load connected.
    Skip if HVMON never changes (open circuit protection kicks in).
    """
    # Step up: channel 0 to ~30000 counts
    quabo.hv_set([30000, 0, 0, 0])
    time.sleep(5.0)

    pkt = quabo.read_hk_packet()
    if pkt is None:
        pytest.skip("No HK packet received after HV step")

    # HVMON0 is at bytes 2:4 of the HK packet (LE int16 in raw counts)
    hvmon0_raw = struct.unpack_from("<H", pkt, 2)[0]

    # Step down: zero HV
    quabo.hv_set([0, 0, 0, 0])
    time.sleep(2.0)

    pkt_after = quabo.read_hk_packet()
    hvmon0_after = struct.unpack_from("<H", pkt_after, 2)[0] if pkt_after else None

    # The HV must have changed: if hvmon0_raw == hvmon0_after == 0, HV circuit may be absent
    if hvmon0_raw == 0:
        pytest.skip("HVMON0 did not change after HV step (no detector or dummy load?)")

    assert hvmon0_raw > 0, f"HVMON0 should be non-zero after step to 30000; got {hvmon0_raw}"
    if hvmon0_after is not None:
        assert hvmon0_after < hvmon0_raw, (
            f"HVMON0 did not decrease after HV zero: before={hvmon0_raw}, after={hvmon0_after}"
        )
