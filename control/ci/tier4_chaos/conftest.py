"""
scenarios/conftest.py

Shared fixtures and helpers for chaos/TDD-forcing scenario tests.

These tests are designed to FAIL RED on current master — they drive
production code rewrites (see plan/panoseti-grpc-c-luminous-toast.md).
Run via: python ci/qa.py chaos
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import uuid
from collections.abc import Iterator
from typing import Any

import pytest

# ── sys.path so we can import control-root modules ────────────────────────────
CONTROL_ROOT = pathlib.Path(__file__).parent.parent.parent.parent.parent

from panoseti_grpc.daq_control.client import DaqControlClient  # noqa: E402

from ci.fixtures.chaos import process_chaos  # noqa: E402
from ci.fixtures.state_probe import StateProbe  # noqa: E402
from ci.tier3_fleet.conftest import (  # noqa: E402
    DAQ_DATA_DIR,
    DAQNODE2_HOST,
    DAQNODE_CONTAINER,
    DAQNODE_DIRECT_HOST,
    GRPC_PORT,
    wait_hashpipe_stopped,
)

# ── Environment ───────────────────────────────────────────────────────────────
INTERLEAVE_PID_FILE = pathlib.Path("tmp/interleave.pid")
MOCK_QUABO_UDS = os.getenv("MOCK_QUABO_UDS", "/tmp/mock_quabo.sock")


# ── Exception types expected from future production code refactors ───────────
#
# These don't exist on master today — that's intentional. The tests that
# reference them will fail with AttributeError/ImportError, proving the
# feature isn't implemented yet.

class StartRunFailed(Exception):
    """Raised by invoke_start_py when start fails (even partially)."""

class StopPartialFailure(Exception):
    """Raised by invoke_stop_py when some (not all) nodes failed to stop."""

class RunAlreadyInProgress(Exception):
    """Raised when a start is attempted while current_run is set."""

class CleanupRefusedPreserveData(Exception):
    """Raised by CleanupData when data preservation is enforced (no force flag)."""

class PHBaselineTooOld(Exception):
    """Raised by start.py when the PH baseline file is more than 24 hours old."""


# ── gRPC API normalizers ──────────────────────────────────────────────────────
#
# The DaqControlClient API:
#   StartDaq(params)  -> bool (True) on success; raises ValueError on failure
#   StopDaq(params)   -> bool (True) on success; raises ValueError on failure
#   StatusDaq(params) -> (bool, dict) tuple
#   CleanupData(params) -> dict {'success': bool, 'message': str}
#
# These helpers normalise all calls to (ok: bool, msg: str) so scenario tests
# can use a consistent pattern without try/except boilerplate everywhere.

def _start(client: DaqControlClient, params: dict[str, Any]) -> tuple[bool, str]:
    """StartDaq → (ok, msg).  Never raises."""
    try:
        client.StartDaq(params)
        return True, ""
    except (ValueError, ConnectionError, Exception) as exc:
        return False, str(exc)


def _stop(client: DaqControlClient, params: dict[str, Any]) -> tuple[bool, str]:
    """StopDaq → (ok, msg).  Never raises."""
    try:
        client.StopDaq(params)
        return True, ""
    except (ValueError, ConnectionError, Exception) as exc:
        return False, str(exc)


def _cleanup(client: DaqControlClient, params: dict[str, Any]) -> tuple[bool, str]:
    """CleanupData dict → (ok, msg).  Never raises."""
    try:
        result = client.CleanupData(params)
        return result.get("success", False), result.get("message", "")
    except (ValueError, ConnectionError, Exception) as exc:
        return False, str(exc)


# ── gRPC invoke helpers ───────────────────────────────────────────────────────
#
# These thin wrappers mimic what start.py/stop.py do at the gRPC layer,
# without requiring the full script to be executable in CI. They let us
# drive the DAQ lifecycle from tests.

def make_run_params(
    data_dir: str = DAQ_DATA_DIR,
    daq_ip_addr: str = DAQNODE_DIRECT_HOST,
    module_ids: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "data_dir": data_dir,
        "daq_ip_addr": daq_ip_addr,
        "bindhost": "lo",
        "max_file_size_mb": 1,
        "group_ph_frames": True,
        "run_dir": f"chaos_run_{uuid.uuid4().hex[:8]}.pffd",
        "obs": "chaos_test",
        "module_id": module_ids or [250, 254],
    }


def grpc_start_daq(
    client: DaqControlClient,
    params: dict[str, Any],
) -> str:
    """Call StartDaq and return the run_dir on success, or raise StartRunFailed."""
    ok, resp = _start(client, params)
    if not ok:
        raise StartRunFailed(f"StartDaq failed: {resp}")
    return params["run_dir"]


def grpc_stop_daq(
    client: DaqControlClient,
    params: dict[str, Any],
    *,
    force_cleanup: bool = False,
) -> None:
    """Call StopDaq + CleanupData. Raises StopPartialFailure on error."""
    ok, resp = _stop(client, {"data_dir": params["data_dir"], "run_dir": params["run_dir"]})
    if not ok:
        raise StopPartialFailure(f"StopDaq failed: {resp}")
    wait_hashpipe_stopped(client, params["data_dir"], timeout=4)

    cleanup_req: dict[str, Any] = {
        "data_dir": params["data_dir"],
        "run_dir": params["run_dir"],
        "module_id": params["module_id"],
    }
    if force_cleanup:
        cleanup_req["force"] = True  # proposed new field — not in proto yet

    ok, resp = _cleanup(client, cleanup_req)
    if not ok:
        if force_cleanup:
            raise StopPartialFailure(f"CleanupData(force=True) failed: {resp}")
        raise CleanupRefusedPreserveData(f"CleanupData refused: {resp}")


def any_pff_files_on_daqnode(run_dir: str, module_ids: list[int] | None = None) -> bool:
    """Check whether any .pff files exist for the run on the shared volume."""
    base = pathlib.Path(DAQ_DATA_DIR)
    mids = module_ids or list(range(256))
    for mid in mids:
        run_path = base / f"module_{mid}" / run_dir
        if run_path.exists() and list(run_path.rglob("*.pff")):
            return True
    return False


@pytest.fixture
def daqnode_fleet(request: Any, docker_client: Any) -> Iterator[Any]:
    """
    Dynamic N-node fleet fixture.
    Usage: @pytest.mark.parametrize("daqnode_fleet", [4], indirect=True)
    """
    from ci.fixtures.fleet import make_fleet
    n_nodes = request.param
    fleet = make_fleet(n_nodes)
    try:
        fleet.start()
        fleet.wait_healthy()
        # fleet.verify_shm() # skip shm check for CI (may fail if shm is not 2g)
        yield fleet
    finally:
        fleet.tear_down()


@pytest.fixture(scope="session")
def daq_control_direct() -> DaqControlClient:
    return DaqControlClient(host=DAQNODE_DIRECT_HOST, port=GRPC_PORT)


@pytest.fixture(scope="session")
def daq_control_node2() -> DaqControlClient:
    return DaqControlClient(host=DAQNODE2_HOST, port=GRPC_PORT)


@pytest.fixture(scope="session")
def daqnode_container() -> Any:
    """Docker container handle for the primary daqnode. Skips if unavailable."""
    try:
        import docker
        client = docker.from_env()
        return client.containers.get(DAQNODE_CONTAINER)
    except Exception as e:
        pytest.skip(f"Docker SDK unavailable or container not found: {e}")


@pytest.fixture
def run_params() -> dict[str, Any]:
    return make_run_params()


@pytest.fixture
def state_probe(daq_control_direct: DaqControlClient) -> StateProbe:
    try:
        import redis
        rc = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            decode_responses=False,
        )
    except Exception:
        rc = None
    loki = os.getenv("LOKI_URL")
    return StateProbe(
        daq_control_client=daq_control_direct,
        redis_client=rc,
        loki_url=loki,
    )


@pytest.fixture
def fresh_run_state(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
) -> Iterator[None]:
    """
    Pre-test: ensure no hashpipe is running.
    Post-test: unconditional stop + cleanup so the next test starts clean.

    Uses StatusDaq(check_run_dirs=True) to discover the currently active run
    directory before calling StopDaq.  The naïve approach of calling StopDaq
    with the current test's (new, not-yet-created) run_dir fails server-side
    validation and leaves a previous test's hashpipe running.
    """
    def _stop_any_running_hashpipe() -> None:
        with contextlib.suppress(Exception):
            _ok, status = daq_control_direct.StatusDaq({
                "data_dir": run_params["data_dir"],
                "check_hashpipe_running": True,
                "check_disk_usage": False,
                "check_run_dirs": True,
            })
            for run_dir_path in status.get("run_dirs", []):
                rdir = run_dir_path.rstrip("/").rsplit("/", 1)[-1]
                with contextlib.suppress(Exception):
                    daq_control_direct.StopDaq({
                        "data_dir": run_params["data_dir"],
                        "run_dir": rdir,
                    })
        wait_hashpipe_stopped(daq_control_direct, run_params["data_dir"], timeout=4)

    # Pre-test cleanup
    _stop_any_running_hashpipe()

    yield

    # Post-test cleanup
    _stop_any_running_hashpipe()
    with contextlib.suppress(Exception):
        daq_control_direct.CleanupData({
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
        })


@pytest.fixture
def kill_hashpipe(daqnode_container: Any) -> Any:
    """Returns a callable(signal='KILL', delay=0) that kills hashpipe."""
    timers: list[Any] = []

    def _kill(signal: str = "KILL", delay: float = 0.0) -> None:
        t = process_chaos.spawn_killer(
            DAQNODE_CONTAINER, "hashpipe", delay_s=delay, sig=signal
        )
        timers.append(t)

    yield _kill

    for t in timers:
        t.cancel()


@pytest.fixture
def mock_quabo_fleet() -> Any:
    """
    Attach to the mock_quabo UDS control socket.
    Skips gracefully if the socket isn't present (mock-quabo not running).
    """
    if not pathlib.Path(MOCK_QUABO_UDS).exists():
        pytest.skip("mock_quabo UDS not found — run with mock-quabo service")

    from ci.mock_quabo.control_client import MockQuaboFleet
    fleet = MockQuaboFleet.attach(uds_path=MOCK_QUABO_UDS)
    try:
        fleet.reset_all()
        yield fleet
    finally:
        fleet.reset_all()
