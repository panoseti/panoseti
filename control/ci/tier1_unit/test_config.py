"""
ci/tier1_unit/test_config.py

Unit tests for PANOSETI configuration models.
Verifies Pydantic strictness, range validation, and cross-field constraints.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from control.topology.fleet import generate_palomar_topology
from control.utils.pydantic_config_models import DaqConfig, DataConfig, ObsConfig

# ---------------------------------------------------------------------------
# DataConfig Constraints
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("integration_time, expected_valid", [
    (10, False),    # too low (min 20)
    (20, True),     # ok
    (100, True),    # ok
    (7, False),     # not multiple of 10
    (7000, False),  # does not divide 1,000,000
    (100000, True), # ok
])
def test_when_integration_time_invalid_then_rejected(integration_time, expected_valid):
    """
    Intent: Verify integration_time_usec constraints (multiple of 10, divisor of 1e6).
    """
    base_data = {
        "run_type": "science",
        "image": {
            "integration_time_usec": integration_time,
            "pe_threshold": 3.0,
            "quabo_sample_size": 8
        }
    }
    if expected_valid:
        DataConfig(**base_data)
    else:
        with pytest.raises(ValidationError):
            DataConfig(**base_data)

@pytest.mark.parametrize("run_type, expected_valid", [
    ("science", True),
    ("engineering", True),
    ("my run", False),         # space not allowed
    ("verylongrunname01", False), # too long
    ("test.run", False),       # dot not allowed
])
def test_when_run_type_malformed_then_rejected(run_type, expected_valid):
    """
    Intent: Verify run_type string constraints (length, invalid chars).
    """
    base_data = {"run_type": run_type}
    if expected_valid:
        DataConfig(**base_data)
    else:
        with pytest.raises(ValidationError):
            DataConfig(**base_data)

def test_when_interleave_missing_keys_then_rejected():
    """
    Intent: Verify that interleave states reference valid top-level mode keys.
    Assertion: Pydantic raises ValidationError when a state references a missing mode.
    """
    cfg = {
        "run_type": "science",
        "interleave": {
            "enable": True,
            "states": [
                {
                    "state_name": "bad_state",
                    "duration_seconds": 2,
                    "movie_mode_config": "image_MISSING",
                    "pulse_height_mode_config": None
                }
            ]
        }
    }
    with pytest.raises(ValidationError, match="references missing movie mode"):
        DataConfig(**cfg)

# ---------------------------------------------------------------------------
# Topology Logic
# ---------------------------------------------------------------------------

def test_when_palomar_generated_then_valid_topology():
    """
    Intent: Ensure the realistic Palomar topology generator produces valid models.
    """
    daq, uids, net, obs = generate_palomar_topology()
    
    # These should all pass internal Pydantic validation
    assert isinstance(daq, DaqConfig)
    assert isinstance(obs, ObsConfig)
    
    # Cross-field check: all modules in obs must be claimed by a DAQ node
    obs_mids = []
    for dome in obs.domes:
        for mod in dome.modules:
            obs_mids.append(mod.id)
            
    daq_mids = []
    for node in daq.daq_nodes:
        daq_mids.extend(node.module_ids)
        
    assert set(obs_mids) == set(daq_mids)
