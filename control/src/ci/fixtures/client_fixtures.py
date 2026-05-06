import logging

import pytest
from panoseti_grpc.daq_control.client import AsyncDaqControlClient, DaqControlClient
from panoseti_grpc.daq_data.client import DaqDataClient

from control.utils import util

from .topology_fixtures import ObservatoryTopology

logger = logging.getLogger("PSETI.CI.Fixtures")

@pytest.fixture
def daq_client(topology: ObservatoryTopology) -> DaqControlClient:
    """Synchronous gRPC client for the primary DAQ node."""
    daq_config = topology._daq
    if not daq_config.daq_nodes:
        raise RuntimeError("No DAQ nodes defined in topology.")
    
    primary = daq_config.daq_nodes[0]
    host, port = util.daq_grpc_endpoint(primary, daq_config)
    logger.debug(f"Resolved primary DAQ endpoint: {host}:{port} (from {primary.ip_addr})")
    return DaqControlClient(host=host, port=port)

@pytest.fixture
def daq_control_direct(daq_client: DaqControlClient) -> DaqControlClient:
    """Alias for daq_client (legacy compat)."""
    return daq_client

@pytest.fixture
def async_daq_client(topology: ObservatoryTopology) -> AsyncDaqControlClient:
    """Asynchronous gRPC client for the primary DAQ node."""
    daq_config = topology._daq
    if not daq_config.daq_nodes:
        raise RuntimeError("No DAQ nodes defined in topology.")
    
    primary = daq_config.daq_nodes[0]
    host, port = util.daq_grpc_endpoint(primary, daq_config)
    return AsyncDaqControlClient(host=host, port=port)

@pytest.fixture
def daq_client_2(topology: ObservatoryTopology) -> DaqControlClient:
    """Synchronous gRPC client for the second DAQ node (if available)."""
    daq_config = topology._daq
    if len(daq_config.daq_nodes) < 2:
        raise RuntimeError("Test requires at least 2 DAQ nodes but only 1 found.")
    
    secondary = daq_config.daq_nodes[1]
    host, port = util.daq_grpc_endpoint(secondary, daq_config)
    return DaqControlClient(host=host, port=port)

@pytest.fixture
def daq_control_node2(daq_client_2: DaqControlClient) -> DaqControlClient:
    """Alias for daq_client_2 (legacy compat)."""
    return daq_client_2

@pytest.fixture
def data_client(topology: ObservatoryTopology) -> DaqDataClient:
    """Synchronous gRPC Data client for the primary DAQ node."""
    daq_config = topology._daq
    if not daq_config.daq_nodes:
        raise RuntimeError("No DAQ nodes defined in topology.")
    
    primary = daq_config.daq_nodes[0]
    # host, port = util.daq_grpc_endpoint(primary, daq_config)
    return DaqDataClient(daq_config.model_dump(), network_config=topology._net.model_dump())
