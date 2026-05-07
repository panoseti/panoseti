# mypy: ignore-errors
"""
test_status_quabo_report.py — Quabo reachability status reporting.

Ported from ci/software_only/tier2_logic/test_status_quabo_report.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from control.start import QuaboProbeResult
from control.status import _sweep_summary


@pytest.mark.asyncio
async def test_status_reports_ok_when_all_up() -> None:
    results = [
        QuaboProbeResult(uid="q1", ip="1.1.1.1", port=60000, reachable=True, error=None),
        QuaboProbeResult(uid="q2", ip="1.1.1.2", port=60000, reachable=True, error=None),
    ]
    
    with patch("control.utils.config_file.get_daq_config"), \
         patch("control.utils.config_file.get_quabo_uids"), \
         patch("control.utils.config_file.get_network_config"), \
         patch("control.utils.util.attach_daq_config"), \
         patch("control.start._check_daq_reachability", new_callable=AsyncMock), \
         patch("control.start._quabo_reachability_report", new_callable=AsyncMock, return_value=results):
        
        lines = await _sweep_summary()
        assert any("Quabos:    OK    — 2/2 reachable" in line for line in lines)

@pytest.mark.asyncio
async def test_status_reports_degraded_when_some_down() -> None:
    results = [
        QuaboProbeResult(uid="q1", ip="1.1.1.1", port=60000, reachable=True, error=None),
        QuaboProbeResult(uid="q2", ip="1.1.1.2", port=60000, reachable=False, error="timeout"),
    ]
    
    with patch("control.utils.config_file.get_daq_config"), \
         patch("control.utils.config_file.get_quabo_uids"), \
         patch("control.utils.config_file.get_network_config"), \
         patch("control.utils.util.attach_daq_config"), \
         patch("control.start._check_daq_reachability", new_callable=AsyncMock), \
         patch("control.start._quabo_reachability_report", new_callable=AsyncMock, return_value=results):
        
        lines = await _sweep_summary()
        assert any("Quabos:    DEGRADED — 1/2 reachable; down: q2" in line for line in lines)

@pytest.mark.asyncio
async def test_status_reports_down_when_all_down() -> None:
    results = [
        QuaboProbeResult(uid="q1", ip="1.1.1.1", port=60000, reachable=False, error=None),
        QuaboProbeResult(uid="q2", ip="1.1.1.2", port=60000, reachable=False, error=None),
    ]
    
    with patch("control.utils.config_file.get_daq_config"), \
         patch("control.utils.config_file.get_quabo_uids"), \
         patch("control.utils.config_file.get_network_config"), \
         patch("control.utils.util.attach_daq_config"), \
         patch("control.start._check_daq_reachability", new_callable=AsyncMock), \
         patch("control.start._quabo_reachability_report", new_callable=AsyncMock, return_value=results):
        
        lines = await _sweep_summary()
        assert any("Quabos:    DOWN  — 0/2 reachable" in line for line in lines)
