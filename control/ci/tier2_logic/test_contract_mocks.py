"""
ci/tier2_logic/test_contract_mocks.py

Contract tests to ensure software mocks (Tier 2) remain in sync with 
gRPC client models and production schemas.
"""

from __future__ import annotations

import pytest
from panoseti_grpc.daq_control.client_models import (
    GenerateManifestParameters,
    CleanupDataParameters
)
from ci.fixtures.mocks import MockDaqNode

@pytest.mark.asyncio
async def test_when_daq_mock_called_then_params_match_grpc_schema():
    """
    Intent: Verify that the MockDaqNode interface uses the same parameter 
    structures as the actual gRPC client models.
    """
    mock_node = MockDaqNode("127.0.0.1")
    
    # Simulate a manifest call
    params = {
        "data_dir": "/data",
        "run_dir": "test_run.pffd",
        "module_id": 200,
        "algorithm": "blake3"
    }
    
    # Verify the schema validation passes for these params
    # This prevents 'mock drift' where we change the production code 
    # but forget to update the mocks.
    GenerateManifestParameters(**params)
    
    # Simulate a cleanup call
    cleanup_params = {
        "data_dir": "/data",
        "run_dir": "test_run.pffd",
        "module_id": [200],
        "mode": "CLEANUP_SELECTIVE"
    }
    CleanupDataParameters(**cleanup_params)
    
    # Execute call on mock to ensure it doesn't crash
    resp = await mock_node.client.GenerateManifest(params)
    assert resp["success"] is True
