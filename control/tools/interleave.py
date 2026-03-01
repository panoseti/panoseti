#!/usr/bin/env python3
"""
interleave.py

PANOSETI Interleaved Observation Controller.
Runs as a background daemon during an active observation to rapidly
switch Quabo FPGA and MAROC registers between different observing modes.
"""

import time
import logging
import argparse
import copy
import sys
import os
import signal
import psutil
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from ..driver import quabo_driver
# import control.config as pano_config
# from control.start import get_daq_params
# from control.stop import stop_data_flow
# from ..utils import config_file, util

from driver import quabo_driver
import config as pano_config
from utils import config_file, util

PID_FILE = "tmp/interleave.pid"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("panoseti.interleave")


class InterleaveController:
    MAX_THREADS = 8
    def __init__(self, data_config: Dict[str, Any], obs_config: Dict[str, Any],
                 daq_config: Dict[str, Any], quabo_uids: Dict[str, Any],
                 quabo_info: List[Dict[str, Any]], network_config: Dict[str, Any],
                 dry_run: bool = False, max_cycles: Optional[int] = None):

        self.keep_running = True
        self.dry_run = dry_run
        self.max_cycles = max_cycles
        self.stats = {"total_cycles": 0, "total_switch_overhead_sec": 0.0}

        self._acquire_lock()

        # Enforce that an active run exists
        run_name = util.read_run_name()
        if not self.dry_run and not run_name:
            self._release_lock()
            logger.error("No run is currently in progress. Interleaving requires an active observation.")
            logger.error("Please run `python start.py` before starting the interleaver.")
            sys.exit(1)
        elif not self.dry_run:
            logger.info(f"Attached to active run: {run_name}")

        self.data_config = copy.deepcopy(data_config)
        self.interleave_cfg = data_config.get("interleave", {})
        self.obs_config = obs_config
        self.daq_config = daq_config
        self.quabo_uids = quabo_uids
        self.quabo_info = quabo_info
        self.network_config = network_config

        self.modules = config_file.get_modules(obs_config)
        self.quabos: Dict[int, List[quabo_driver.QUABO]] = {}  # map module_id to quabo_driver instances
        for module in self.modules:
            base_ip_addr = module['ip_addr']
            module_id = config_file.ip_addr_to_module_id(base_ip_addr)
            self.quabos[module_id] = []
            for i in range(4):
                uid = util.quabo_uid(module, quabo_uids, i)
                if not uid: continue
                ip_addr = config_file.quabo_ip_addr(base_ip_addr, i)
                ip_ports = util.get_quabo_ip_port(base_ip_addr, i, network_config)
                real_ip = ip_ports['ip_addr']
                cmd_port = ip_ports['cmd_port']
                q = quabo_driver.QUABO(real_ip, cmd_port)
                self.quabos[module_id].append(q) # use a list to retain sequential ordering

        self.executor = ThreadPoolExecutor(max_workers=self.MAX_THREADS)

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)

        if self.dry_run:
            logger.info("=== DRY RUN MODE ENABLED: Hardware commands will be simulated ===")

    def _acquire_lock(self):
        """Ensures at most one instance of interleave.py is running."""
        if os.path.exists(PID_FILE):
            try:
                with open(PID_FILE, "r") as f:
                    old_pid = int(f.read().strip())

                if psutil.pid_exists(old_pid):
                    logger.critical(
                        f"CRITICAL: Another interleave process (PID {old_pid}) is currently running.\n"
                        "To resolve this, run `python config.py --stop-interleave`."
                    )
                    sys.exit(1)
                else:
                    logger.warning(f"Stale PID file detected for dead process {old_pid}. Cleaning up...")
                    os.remove(PID_FILE)
            except (ValueError, OSError):
                os.remove(PID_FILE)

        os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))

    def _release_lock(self):
        if os.path.exists(PID_FILE):
            try:
                os.remove(PID_FILE)
            except OSError:
                pass

    def _handle_shutdown_signal(self, signum, frame):
        if self.keep_running:
            logger.warning("Shutdown signal received. Breaking cycle to restore defaults...")
            self.keep_running = False

    def _broadcast_acq_mode(self, daq_params: quabo_driver.DAQ_PARAMS) -> None:
        """Broadcast the ACQ mode:
            - In parallel to all modules but
            - Sequentially to quabos in the same module
        """
        if self.dry_run:
            logger.info(f"[DRY-RUN] Simulating ACQ broadcast: do_image={daq_params.do_image}, do_ph={daq_params.do_ph}")
            return

        def send_acq_mode_to_module(module_id: int):
            """Sequentially broadcast ACQ mode to Quabos in the Q0, Q1, Q2, Q3 order."""
            quabos = self.quabos[module_id]
            for q in quabos:
                q.send_daq_params(daq_params)
        futures = []
        for mid in self.quabos.keys():
            futures.append(self.executor.submit(send_acq_mode_to_module, mid))
        for f in as_completed(futures):
            f.result()

    def _reconfigure_quabos(self, next_state_data_config: Dict[str, Any]) -> None:
        if self.dry_run:
            return

        def reconfig_module(module):
            pano_config.do_maroc_config(
                [module], self.quabo_uids, self.quabo_info,
                next_state_data_config, self.obs_config, self.daq_config,
                self.network_config, verbose=False, write_config=False
            )
            pano_config.do_mask_config(
                [module], next_state_data_config, self.network_config,
                self.quabo_uids, verbose=False, write_config=False, do_flush_rx_buf=False,
            )

        futures = [self.executor.submit(reconfig_module, module) for module in self.modules]
        for f in as_completed(futures): f.result()

    def generate_state_dict(self, movie_key: Optional[str], ph_key: Optional[str]) -> Dict[str, Any]:
        temp_dict = self.data_config.copy()
        temp_dict.pop('image', None)
        temp_dict.pop('pulse_height', None)
        if movie_key: temp_dict['image'] = self.data_config[movie_key]
        if ph_key: temp_dict['pulse_height'] = self.data_config[ph_key]
        logger.debug(f"Generating state dict for {temp_dict=}")
        return temp_dict

    def _sleep_until(self, target_time: float, spin_wait_threshold: float = 0.005) -> None:
        while self.keep_running:
            now = time.perf_counter()
            remaining = target_time - now
            if remaining <= spin_wait_threshold: break
            time.sleep(remaining - spin_wait_threshold)
        while self.keep_running and time.perf_counter() < target_time:
            pass


    def run_loop(self) -> None:
        if not self.interleave_cfg.get("enable", False):
            logger.info("Interleaving disabled in config. Exiting.")
            self._release_lock()
            return

        states = self.interleave_cfg.get("states", [])
        stop_daq_params = quabo_driver.DAQ_PARAMS(False, 0, False, False, False)

        try:
            # Must import start dynamically to avoid circular import errors
            from start import get_daq_params
            while self.keep_running:
                if self.max_cycles and self.stats["total_cycles"] >= self.max_cycles:
                    logger.info(f"Max cycles ({self.max_cycles}) reached. Ending run_loop.")
                    break

                for state in states:
                    if not self.keep_running: break

                    name = state["state_name"]
                    duration = state["duration_seconds"]

                    # make the next interleave state appear as the "default" state
                    # this lets us use existing production helper functions to config
                    next_state_data_config = self.generate_state_dict(
                        state.get("movie_mode_config"),
                        state.get("pulse_height_mode_config")
                    )
                    start_daq_params = get_daq_params(next_state_data_config)

                    logger.info(f"\n--- Entering State: {name} (Duration: {duration}s) ---")
                    t_overhead_start = time.perf_counter()

                    # 1. Stop data flow
                    self._broadcast_acq_mode(stop_daq_params)
                    # 2. Reconfigure Quabos
                    self._reconfigure_quabos(next_state_data_config)
                    # 3. Start data flow
                    self._broadcast_acq_mode(start_daq_params)

                    overhead = time.perf_counter() - t_overhead_start
                    self.stats["total_switch_overhead_sec"] += overhead

                    logger.info(f"Hardware configured in {overhead:.2f}s. Actively observing for {duration}s...")

                    # Relative Active Scheduling: Guarantee the full duration occurs AFTER configuration
                    self._sleep_until(time.perf_counter() + duration)

                self.stats["total_cycles"] += 1

        except Exception as e:
            logger.error(f"Error in interleaving loop: {e}", exc_info=True)
        finally:
            self._teardown()

    def _teardown(self):
        """Restores Quabos to default settings and cleans up."""
        if self.dry_run:
            logger.info("[DRY-RUN] Teardown initiated. Simulating hardware default restoration.")
            self._release_lock()
            return

        logger.info("Teardown initiated. Restoring default hardware configuration...")
        stop_daq_params = quabo_driver.DAQ_PARAMS(False, 0, False, False, False)
        try:
            from start import get_daq_params
            # get default daq params
            next_state_data_config = self.data_config
            start_default_daq_params = get_daq_params(next_state_data_config)

            # Forcefully drop old tasks and spin up a fresh pool to guarantee teardown commands aren't stuck behind deadlocked threads.
            self.executor.shutdown(wait=False)
            self.executor = ThreadPoolExecutor(max_workers=self.MAX_THREADS)

            # restore default parameters
            self._broadcast_acq_mode(stop_daq_params)
            self._reconfigure_quabos(next_state_data_config)
            self._broadcast_acq_mode(start_default_daq_params)

            logger.info("Hardware defaults restored successfully.")
        except Exception as e:
            logger.exception(f"Failed to restore hardware defaults: {e}")
        finally:
            self.executor.shutdown(wait=False)
            for quabos in self.quabos.values():
                for q in quabos: q.close()
            self._release_lock()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PANOSETI Interleave Controller")
    parser.add_argument('--dry-run', action='store_true', help='Simulate execution without hardware commands')
    parser.add_argument('--max-cycles', type=int, default=None, help='Limit the number of schedule loops')
    args = parser.parse_args()

    controller = None
    try:
        controller = InterleaveController(
            data_config=config_file.get_data_config(),
            obs_config=config_file.get_obs_config(),
            daq_config=config_file.get_daq_config(),
            quabo_uids=config_file.get_quabo_uids(),
            quabo_info=config_file.get_quabo_info(),
            network_config=config_file.get_network_config(),
            dry_run=args.dry_run,
            max_cycles=args.max_cycles
        )
        controller.run_loop()
    except Exception as e:
        logger.error(f"Interleave startup failed: {e}")
        sys.exit(1)