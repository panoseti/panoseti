"""
conftest for core_happy_path tests.

Provides two key fixtures:
  booted_calibrated  — session-scoped; ensures hardware is in PH_CALIBRATED.
  active_data_config — function-scoped, parameterized; swaps data_config.json
                       symlink and re-applies maroc_config + mask_config.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ci.hardware_software.core.reachability import wait_until_all_quabos_reachable

logger = logging.getLogger(__name__)

# Paths used by the data-config swap fixture.
# CONFIGS is where pseti reads data_config.json (set via $PSETI_CONFIG).
# CORE_OBS_CONFIGS is where our variant JSON files live.
_HW_SW_DIR = Path(__file__).parent.parent
CONFIGS = _HW_SW_DIR / "configs"
CORE_OBS_CONFIGS = _HW_SW_DIR / "core_obs_configs"

# Ordered list of data-config variant names to parametrize.
# Add new entries here when more variants are validated.
DATA_CONFIGS = [
    "image_8bit",
    "pulse_height_uhe",
    "interleave",
]


# ---------------------------------------------------------------------------
# booted_calibrated — session-scoped boot gate
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def booted_calibrated(runner, topology):
    """Ensure hardware is in PH_CALIBRATED before any happy-path test runs.

    If the HITL state file already records PH_CALIBRATED and all quabos are
    responsive (e.g., from a prior manual session-start), the boot sequence
    is skipped for speed.  Otherwise the full boot sequence is run once.

    If the boot fails, the fixture raises and all happy_path tests are
    cascade-skipped by pytest_plugin.py.
    """
    from ci.hardware_software.core.boot import quabos_responsive, state_file_says
    from ci.hardware_software.hw_utils.cli import _STATE_FILE
    from ci.hardware_software.hw_utils.state_machine import _write_state

    if state_file_says("PH_CALIBRATED") and quabos_responsive():
        logger.info("[HAPPY-PATH] booted_calibrated: hardware already in PH_CALIBRATED — skipping boot")
        yield
        return

    logger.info("[HAPPY-PATH] booted_calibrated: running full boot sequence")
    import os
    import time

    import control.config as config
    import control.get_uids as get_uids
    from control.pseti import app
    from control.utils import config_file

    obs_config = config_file.get_obs_config()
    network_config = config_file.get_network_config()
    modules = config_file.get_modules(obs_config)

    # Stage 0: power off for clean baseline
    r = runner.invoke(app, ["power", "off"])
    assert r.exit_code == 0, f"booted_calibrated: pseti power off failed:\n{r.output}"
    time.sleep(5)

    # Stage 1: power on
    from ci.hardware_software.hw_utils.driver_ops import wps_power_on
    wps_power_on()

    # Stage 2: boot wait
    boot_wait = int(os.environ.get("HW_TEST_QUABO_BOOT_WAIT", 60))
    logger.info("booted_calibrated: waiting %ds for boot", boot_wait)
    elapsed = 0
    while elapsed < boot_wait:
        chunk = min(15, boot_wait - elapsed)
        time.sleep(chunk)
        elapsed += chunk

    # Stage 3-4: UIDs + do_reboot
    get_uids.get_uids(obs_config, network_config)
    quabo_uids = config_file.get_quabo_uids()
    config.do_reboot(modules, quabo_uids, network_config)
    _write_state(_STATE_FILE, "BOOTED")

    # Stage 6-10: hk-dest → redis-daemons → maroc-config → mask-config → calibrate-ph
    for _stage, args in [
        ("hk-dest",       ["cfg", "hk-dest"]),
        ("redis-daemons", ["cfg", "redis-daemons"]),
        ("maroc-config",  ["cfg", "maroc-config"]),
        ("mask-config",   ["cfg", "mask-config"]),
        ("calibrate-ph",  ["cfg", "calibrate-ph"]),
    ]:
        # maroc-config may prompt "Use default calibration file?" for each quabo
        # whose UID isn't in quabo_info.json — auto-accept with Y.
        stdin = "Y\nY\nY\nY\n" if "maroc-config" in args else None
        r = runner.invoke(app, args, input=stdin)
        assert r.exit_code == 0, f"booted_calibrated: pseti {' '.join(args)} failed:\n{r.output}"

    _write_state(_STATE_FILE, "PH_CALIBRATED")
    logger.info("booted_calibrated: hardware in PH_CALIBRATED")
    wait_until_all_quabos_reachable(topology, timeout=30, retry_every=2)
    yield


# ---------------------------------------------------------------------------
# active_data_config — parameterized symlink swap fixture
# ---------------------------------------------------------------------------

@pytest.fixture(params=DATA_CONFIGS)
def active_data_config(request, runner, topology):
    """Point configs/data_config.json at a specific variant, re-apply maroc+mask.

    Parameterized over DATA_CONFIGS so each variant gets its own test run.
    No power cycle is needed between variants — only maroc_config and
    mask_config need re-applying to reload gains/masks from the new config.
    """
    name = request.param
    src = CORE_OBS_CONFIGS / f"data_config_{name}.json"
    dst = CONFIGS / "data_config.json"

    if not src.exists():
        pytest.skip(f"Data config variant not found: {src}")

    # Swap the symlink
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())
    logger.info("active_data_config: configs/data_config.json → %s", src.name)

    # Re-apply hardware config for this variant
    from control.pseti import app
    r = runner.invoke(app, ["cfg", "maroc-config"], input="Y\nY\nY\nY\n")
    assert r.exit_code == 0, f"active_data_config({name}): pseti cfg maroc-config failed:\n{r.output}"
    r = runner.invoke(app, ["cfg", "mask-config"])
    assert r.exit_code == 0, f"active_data_config({name}): pseti cfg mask-config failed:\n{r.output}"

    wait_until_all_quabos_reachable(topology, timeout=30, retry_every=2)
    yield name

    # Restore symlink to image_8bit as a safe default
    default_src = CORE_OBS_CONFIGS / "data_config_image_8bit.json"
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if default_src.exists():
        dst.symlink_to(default_src.resolve())
        logger.info("active_data_config: restored configs/data_config.json → data_config_image_8bit.json")
