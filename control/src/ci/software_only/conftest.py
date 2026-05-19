"""
conftest.py — Software-only fixtures for the panoseti-control test suite.
Extends the shared fixtures in ci/conftest.py with Docker-CI isolation.
"""

import contextlib
import os
import pathlib
import shutil
import time
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from panoseti_grpc.daq_control.client import DaqControlClient
from panoseti_grpc.daq_data.client import DaqDataClient

from ci.fixtures.fleet import Fleet
from ci.paths import PanoPathsTest

pytest_plugins = [
    "ci.fixtures.workspace_fixtures",
    "ci.fixtures.network_fixtures",
    "ci.fixtures.data_fixtures",
    "ci.fixtures.topology_fixtures",
    "ci.fixtures.client_fixtures",
    "ci.fixtures.rsync_fixtures",
    "ci.fixtures.transfer_fixtures",
    "ci.fixtures.chaos_fixtures",
    "ci.fixtures.fleet",
    "ci.fixtures.state_probe",
    "ci.fixtures.mocks",
]


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
        
    # 2. Populate static configs from the known-good direct/ config base.
    # We always seed from PanoPathsTest.integration_configs("direct") rather than from
    # $PSETI_CONFIG, because $PSETI_CONFIG may point to hardware_software/configs which
    # uses Docker-only symlinks (e.g. data_config.json → /app/...) that are broken on
    # the host. The topological configs (daq/obs/network/quabo_uids) are regenerated
    # fresh below, so we only copy the static non-topological files here.
    from ci.paths import PanoPathsTest as _PanoPathsTest
    _static_src = _PanoPathsTest.integration_configs("direct")
    head_prefix = os.environ.get("HEAD_NET_PREFIX", "10.0.1")
    daq_prefix = os.environ.get("DAQ_NET_PREFIX", "192.168.0")
    quabo_prefix = os.environ.get("QUABO_NET_PREFIX", "192.168.3")

    _topological = {"daq_config.json", "obs_config.json", "network_config.json", "quabo_uids.json"}
    if _static_src.exists():
        for item in _static_src.iterdir():
            if item.name in _topological:
                continue
            try:
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
    print("THE TMP PATH:", str(tmp_path))
    

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

    # Overwrite the configs written by auto_isolate with the real fleet details
    cfg_dir = pathlib.Path(os.environ["PSETI_CONFIG"])
    fleet.write_daq_config(cfg_dir / "daq_config.json", head_node_ip=f"{head_prefix}.22")
    
    # Also update network_config.json to ensure no stale PF data exists
    (cfg_dir / "network_config.json").write_text('{"modules": [], "daq_nodes": []}')

    # Invalidate config cache
    import importlib

    from control.utils import config_file
    importlib.reload(config_file)

    os.environ["REDIS_HOST"] = "localhost"
    os.environ["REDIS_PORT"] = str(fleet.redis_port)
    os.environ["LOKI_URL"] = f"http://localhost:{fleet.loki_port}"

    yield fleet, daq_cfg
    fleet.tear_down()

@pytest.fixture(scope="session")
def topology(session_fleet):
    """Universal topology source of truth, pinned to the session fleet."""
    from ci.fixtures.topology_fixtures import ObservatoryTopology
    return ObservatoryTopology()

@pytest.fixture(scope="session")
def daqnode_container(session_fleet) -> Any:
    fleet, _ = session_fleet
    return fleet.containers[0].get_wrapped_container()

@pytest.fixture(scope="session")
def daq_control_direct(session_fleet):
    from panoseti_grpc.daq_control.client import DaqControlClient
    fleet, _daq_cfg = session_fleet
    spec = fleet.specs[0]
    return DaqControlClient(host=spec.container_host_ip, port=spec.mapped_port)

@pytest.fixture(scope="session")
def daq_control_gateway(session_fleet):
    from panoseti_grpc.daq_control.client import DaqControlClient
    fleet, _daq_cfg = session_fleet
    spec = fleet.specs[0]
    return DaqControlClient(host=spec.container_host_ip, port=spec.mapped_port)

@pytest.fixture
def daq_client(daq_control_direct: DaqControlClient) -> DaqControlClient:
    """Fleet override: routes to the real container endpoint (specs[0])."""
    return daq_control_direct

@pytest.fixture(scope="session")
def _daq_control_node2_session(session_fleet):
    from panoseti_grpc.daq_control.client import DaqControlClient
    fleet, _daq_cfg = session_fleet
    spec = fleet.specs[1]
    return DaqControlClient(host=spec.container_host_ip, port=spec.mapped_port)

@pytest.fixture
def daq_control_node2(_daq_control_node2_session: DaqControlClient) -> DaqControlClient:
    """Fleet override: routes node-2 client to the real container endpoint (specs[1])."""
    return _daq_control_node2_session

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
    fleet, _daq_cfg = session_fleet
    spec = fleet.specs[0]
    with DaqDataClient(host=spec.container_host_ip, port=spec.mapped_port) as client:
        yield client

@pytest.fixture
def head_data_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Per-test head node data directory, isolated via tmp_path."""
    d = tmp_path / "head_data"
    d.mkdir(parents=True, exist_ok=True)
    return d

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


# ---------------------------------------------------------------------------
# Polling helpers — replace time.sleep with condition-based waits
# ---------------------------------------------------------------------------

def wait_until(
    condition: Callable[[], bool],
    *,
    timeout: float = 10.0,
    interval: float = 0.2,
) -> bool:
    """Poll condition() until it returns truthy or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if condition():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def wait_hashpipe_running(
    client: DaqControlClient,
    data_dir: str,
    *,
    timeout: float = 10.0,
) -> bool:
    """Poll StatusDaq until hashpipe_running=True or timeout."""
    params = {
        "data_dir":               data_dir,
        "check_hashpipe_running": True,
        "check_disk_usage":       False,
        "check_run_dirs":         False,
    }
    return wait_until(
        lambda: client.StatusDaq(params, timeout=2.0)[1].get("hashpipe_running") is True,
        timeout=timeout,
    )


def wait_hashpipe_stopped(
    client: DaqControlClient,
    data_dir: str,
    *,
    timeout: float = 10.0,
) -> bool:
    """Poll StatusDaq until hashpipe_running is False/None or timeout."""
    params = {
        "data_dir":               data_dir,
        "check_hashpipe_running": True,
        "check_disk_usage":       False,
        "check_run_dirs":         False,
    }
    return wait_until(
        lambda: client.StatusDaq(params, timeout=2.0)[1].get("hashpipe_running") is not True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Real Hashpipe Test module-scoped fixture: start hashpipe + tcpreplay
# ---------------------------------------------------------------------------

# Path to PCAP file *inside* the daqnode container
PCAP_CONTAINER_DIR = "/app/src/ci/fixtures/data/"
PCAP_GLOB = "*.pcapng"

# hp_io_cfg for real (non-simulated) hashpipe mode
REAL_HP_IO_CFG = {
    "update_interval_seconds": 0.1,
    "simulate_daq": False,
    "force": True,
    "module_ids": [],   # stream from all active modules
}

HASHPIPE_READY_RETRIES = 20


@pytest.fixture(scope="module")
def hashpipe_pcap_session(daqnode_container: Any, daq_control_direct: DaqControlClient, run_params: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """
    Start hashpipe via daq_control gRPC, inject PCAP packets via docker exec
    tcpreplay, then yield. Tears down hashpipe on exit.
    """
    # 0. Verify PCAP exists so tcpreplay doesn't silently fail.
    # We check relative to PCAP_CONTAINER_DIR
    res = daqnode_container.exec_run(f"sh -c 'ls {PCAP_CONTAINER_DIR}/{PCAP_GLOB}'")
    if res.exit_code != 0:
        pytest.skip(f"PCAP missing in container at {PCAP_CONTAINER_DIR}/{PCAP_GLOB} — real-data tests require the Docker Compose daqnode with /app mounted")
    
    # 1. Start hashpipe via gRPC (bindhost=lo so it receives loopback packets)
    assert run_params.get("bindhost", "lo") == "lo", f"run_params must have bindhost='lo' for tcpreplay command to stream datat to hashpipe: {run_params=}"
    lp = {**run_params, "bindhost": "lo"}
    try:
        daq_control_direct.StartDaq(lp)
    except Exception as e:
        pytest.fail(f"Failed to start hashpipe via gRPC: {e}")

    # 2. Wait for hashpipe to be confirmed running
    if not wait_hashpipe_running(
        daq_control_direct, run_params["data_dir"], timeout=HASHPIPE_READY_RETRIES
    ):
        pytest.fail(f"hashpipe did not start within {HASHPIPE_READY_RETRIES}s")
    
    # Forces the native Linux veth to accept the foreign MAC addresses from the PCAP
    daqnode_container.exec_run("ip link set lo promisc on")

    # 3. Run tcpreplay inside daqnode container
    replay_cmd = f"sh -c 'tcpreplay --mbps=0.1 --loop=0 --intf1=lo {PCAP_GLOB}'"
    _exit_code, _output = daqnode_container.exec_run(
        replay_cmd,
        detach=True,
        workdir=PCAP_CONTAINER_DIR
    )

    yield run_params
    
    # 4. Teardown
    daqnode_container.exec_run("pkill -9 tcpreplay", detach=False)

    with contextlib.suppress(Exception):
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
    assert wait_hashpipe_stopped(daq_control_direct, run_params["data_dir"], timeout=8)


# ---------------------------------------------------------------------------
# Clean-up Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def ensure_clean_daq_state(daq_control_direct, daq_control_node2) -> Iterator[None]:
    """Ensure no Hashpipe instances are running before and after the test."""
    def _stop_all():
        for client in (daq_control_direct, daq_control_node2):
            with contextlib.suppress(Exception):
                # Use a long timeout to allow the server's 60s graceful wait to complete
                client.StopDaq({"data_dir": "/data", "run_dir": ""}, timeout=70.0)
            wait_hashpipe_stopped(client, "/data", timeout=10)
        from control.utils.run_state import RunStateManager
        RunStateManager().clear_state()

    _stop_all()
    yield
    _stop_all()
