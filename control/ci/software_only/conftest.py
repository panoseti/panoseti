"""
conftest.py — Software-only fixtures for the panoseti-control test suite.
Extends the shared fixtures in ci/conftest.py with Docker-CI isolation.
"""

import os
import pathlib
import shutil
import uuid
import time
from collections.abc import Iterator
from typing import Any

import pytest
from panoseti_grpc.daq_control.client import DaqControlClient
from panoseti_grpc.daq_data.client import DaqDataClient

from ci.fixtures.fleet import Fleet
from ci.paths import PanoPathsTest

def pytest_configure(config: Any) -> None:
    """
    Set environment variable overrides to isolate the test environment.
    This ensures PanoPaths resolves to test-specific directories instead of
    production code directories, preventing state leakage.
    """
    # 1. Route configs to the integration test configs (default to direct for unit tests)
    if "PSETI_CONFIG" not in os.environ:
        os.environ["PSETI_CONFIG"] = str(PanoPathsTest.integration_configs("direct"))

    # 2. Route state to isolated test directories (fallback defaults)
    if "PSETI_TMP" not in os.environ:
        os.environ["PSETI_TMP"] = "/tmp/pseti_test/tmp"
    if "PSETI_LOGS" not in os.environ:
        os.environ["PSETI_LOGS"] = "/tmp/pseti_test/logs"
    if "PSETI_QUABOS" not in os.environ:
        os.environ["PSETI_QUABOS"] = "/tmp/pseti_test/quabos"

    # 3. Ensure directories exist
    os.makedirs(os.environ["PSETI_TMP"], exist_ok=True)
    os.makedirs(os.environ["PSETI_LOGS"], exist_ok=True)
    os.makedirs(os.environ["PSETI_QUABOS"], exist_ok=True)

    # 4. Give each xdist worker its own testcontainers Ryuk session so parallel
    #    workers don't collide on the shared TC_SESSION_ID (409 Conflict).
    import uuid as _uuid
    if hasattr(config, "workerinput"):
        # xdist worker process
        worker_id = config.workerinput.get("workerid", "master")
        run_uuid = config.workerinput.get("tc_run_uuid", _uuid.uuid4().hex[:8])
    else:
        # single-process run (fleet/chaos suites run without xdist)
        worker_id = "solo"
        run_uuid = _uuid.uuid4().hex[:8]

    os.environ["TC_SESSION_ID"] = f"tc-{worker_id}-{run_uuid}"


@pytest.fixture(scope="session", autouse=True)
def auto_isolate(
    tmp_path_factory: pytest.TempPathFactory, 
    worker_id: str
) -> Iterator[pathlib.Path]:
    """
    Autouse session-scoped fixture that provides isolation for configs, transient state,
    and telemetry databases.
    """
    tmp_path = tmp_path_factory.mktemp(f"session_{worker_id}")

    # 1. Setup isolated directories inside tmp_path
    cfg_tmp = tmp_path / "configs"
    state_tmp = tmp_path / "state"
    ctl_tmp = tmp_path / "control"
    tmp_tmp = tmp_path / "tmp"
    
    for d in [cfg_tmp, state_tmp, ctl_tmp, tmp_tmp]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 2. Populate configs from current PSETI_CONFIG
    src_cfg = os.environ.get("PSETI_CONFIG")
    head_prefix = os.environ.get("HEAD_NET_PREFIX", "10.0.1")
    daq_prefix = os.environ.get("DAQ_NET_PREFIX", "192.168.0")
    quabo_prefix = os.environ.get("QUABO_NET_PREFIX", "192.168.3")

    # In software-only conftest, we ASSUME we are NOT a HW-SW test
    is_hw_sw_test = False

    if src_cfg and os.path.exists(src_cfg):
        for item in pathlib.Path(src_cfg).iterdir():
            try:
                # Copy everything EXCEPT the topological configs we are about to overwrite
                if item.name not in ["daq_config.json", "obs_config.json", "network_config.json", "quabo_uids.json"]:
                    if item.is_file():
                        shutil.copy2(item, cfg_tmp)
                    elif item.is_dir():
                        shutil.copytree(item, cfg_tmp / item.name, dirs_exist_ok=True)
            except Exception:
                pass
                
    # 2.5 Generate Pristine Topological Configs
    from control.topology.fleet import generate_ci_topology
    daq_cfg, quabo_uids, net_cfg, obs_cfg = generate_ci_topology(head_prefix, daq_prefix, quabo_prefix)
    
    (cfg_tmp / "daq_config.json").write_text(daq_cfg.model_dump_json(indent=2))
    (cfg_tmp / "obs_config.json").write_text(obs_cfg.model_dump_json(indent=2))
    (cfg_tmp / "network_config.json").write_text(net_cfg.model_dump_json(indent=2))
    
    # 4. Provide quabo_uids.json for Chaos tests
    uids_src = pathlib.Path(__file__).parent.parent / "fixtures" / "configs" / "quabo_uids_chaos.json"
    if uids_src.exists() and os.environ.get("PSETI_TEST_TIER") == "tier4_chaos":
        shutil.copy(uids_src, cfg_tmp / "quabo_uids.json")
        shutil.copy(uids_src, tmp_tmp / "quabo_uids.json")
        os.chmod(cfg_tmp / "quabo_uids.json", 0o666)
        os.chmod(tmp_tmp / "quabo_uids.json", 0o666)
    else:
        (cfg_tmp / "quabo_uids.json").write_text(quabo_uids.model_dump_json(indent=2))
        (tmp_tmp / "quabo_uids.json").write_text(quabo_uids.model_dump_json(indent=2))

    # 3. Apply overrides for the duration of the session
    os.environ["PSETI_CONFIG"] = str(cfg_tmp)
    os.environ["PSETI_STATE"] = str(state_tmp)
    os.environ["PSETI_CONTROL"] = str(ctl_tmp)
    os.environ["PSETI_TMP"] = str(tmp_tmp)
    os.environ["PSETI_QUABOS"] = str(tmp_tmp)

    # Expose isolated data dirs
    if "HEAD_DATA_DIR" not in os.environ:
        head_data_tmp = tmp_path / "head_data"
        head_data_tmp.mkdir(parents=True, exist_ok=True)
        os.environ["HEAD_DATA_DIR"] = str(head_data_tmp)
        os.chmod(str(head_data_tmp), 0o777)

    if "DAQ_DATA_DIR" not in os.environ:
        daq_data_tmp = tmp_path / "daq_data"
        daq_data_tmp.mkdir(parents=True, exist_ok=True)
        os.environ["DAQ_DATA_DIR"] = str(daq_data_tmp)
        os.chmod(str(daq_data_tmp), 0o777)

    # 5. Refresh Pydantic's perspective of the environment
    import importlib
    from control.utils import config_file
    importlib.reload(config_file)

    # 4. Telemetry and Database Isolation
    try:
        db_index = int("".join(filter(str.isdigit, worker_id))) if worker_id != "master" else 0
    except ValueError:
        db_index = 0
        
    os.environ["REDIS_DB"] = str(db_index)
    os.environ["LOKI_TENANT_ID"] = f"test_tenant_{db_index}"
    
    # 5. Ensure PanoPaths and RunStateManager are fresh
    from control.utils.paths import PanoPaths
    from control.utils.run_state import RunStateManager
    PanoPaths.ensure_state_dirs()
    RunStateManager().clear_state()
    
    yield tmp_path


@pytest.fixture(scope="session")
def session_fleet(auto_isolate) -> Iterator[tuple[Fleet, dict[str, Any]]]:
    """Start a 2-node testcontainers fleet and yield (fleet, daq_cfg_dict)."""
    from ci.fixtures.fleet import setup_docker_host
    from control.utils.config_file import ip_addr_to_module_id

    setup_docker_host()

    quabo_prefix = os.environ.get("QUABO_NET_PREFIX", "192.168.3")
    mid1 = ip_addr_to_module_id(f"{quabo_prefix}.32")
    mid2 = ip_addr_to_module_id(f"{quabo_prefix}.36")
    
    from ci.fixtures.fleet import DaqnodeSpec, Fleet
    tc_id = os.environ.get("TC_SESSION_ID", "solo")
    specs = [
        DaqnodeSpec(name=f"pseti-daqnode-{tc_id}-0", module_ids=[mid1]),
        DaqnodeSpec(name=f"pseti-daqnode-{tc_id}-1", module_ids=[mid2]),
    ]
    fleet = Fleet(specs)
    
    try:
        fleet.start()
        fleet.wait_healthy(timeout=90.0)
    except Exception as exc:
        fleet.tear_down()
        raise RuntimeError(f"Fleet failed to start or become healthy: {exc}") from exc

    head_prefix = os.environ.get("HEAD_NET_PREFIX", "10.0.1")
    daq_config = fleet.to_daq_config(head_node_ip=f"{head_prefix}.22")
    daq_cfg: dict[str, Any] = daq_config.model_dump()

    cfg_dir = pathlib.Path(os.environ["PSETI_CONFIG"])
    daq_config_path = cfg_dir / "daq_config.json"
    fleet.write_daq_config(daq_config_path)

    os.environ["REDIS_HOST"] = "localhost"
    os.environ["REDIS_PORT"] = str(fleet.redis_port)
    os.environ["LOKI_URL"] = f"http://localhost:{fleet.loki_port}"

    yield fleet, daq_cfg
    fleet.tear_down()

@pytest.fixture(scope="session")
def daq_control_direct(session_fleet) -> DaqControlClient:
    fleet, _daq_cfg = session_fleet
    spec = fleet.specs[0]
    return DaqControlClient(host=spec.container_host_ip, port=spec.mapped_port)

@pytest.fixture(scope="session")
def daq_control_node2(session_fleet) -> DaqControlClient:
    fleet, _daq_cfg = session_fleet
    spec = fleet.specs[1]
    return DaqControlClient(host=spec.container_host_ip, port=spec.mapped_port)

@pytest.fixture(scope="session")
def daqnode_container(session_fleet) -> Any:
    fleet, _ = session_fleet
    return fleet.containers[0].get_wrapped_container()

@pytest.fixture(scope="session")
def daq_control_gateway(session_fleet) -> DaqControlClient:
    fleet, _daq_cfg = session_fleet
    spec = fleet.specs[0]
    return DaqControlClient(host=spec.container_host_ip, port=spec.mapped_port)

@pytest.fixture(scope="session")
def redis_client(session_fleet) -> Iterator[Any]:
    import redis
    fleet, _daq_cfg = session_fleet
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=fleet.redis_port,
        db=int(os.getenv("REDIS_DB", "0")),
        decode_responses=True
    )
    yield r

@pytest.fixture(scope="session")
def daq_data_client(session_fleet) -> Iterator[DaqDataClient]:
    _fleet, daq_cfg = session_fleet
    with DaqDataClient(daq_cfg, network_config=None) as client:
        yield client

@pytest.fixture(scope='module')
def run_params(session_fleet) -> dict[str, Any]:
    fleet, _ = session_fleet
    return {
        "data_dir":         "/data",
        "daq_ip_addr":      fleet.node_ip(0),
        "bindhost":         "lo",
        "max_file_size_mb": 1,
        "group_ph_frames":  True,
        "run_dir":          f"ci_run_{uuid.uuid4().hex[:8]}.pffd",
        "obs":              "citest",
        "module_id":        [250, 254],
    }
