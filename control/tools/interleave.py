#!/usr/bin/env python3
"""
interleave.py

PANOSETI Interleaved Observation Controller.
Runs as a background daemon during an active observation to rapidly
switch Quabo FPGA and MAROC registers between different observing modes.
"""

import argparse
import contextlib
import logging
import os
import signal
import sys
import time
from typing import Any

import numpy as np
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config as pano_config
from driver import quabo_driver
from utils import config_file, util
from utils.pydantic_config_models import (
    DaqConfigValidator,
    DataConfigValidator,
    InterleaveConfig,
    NetworkConfigValidator,
    ObsConfigValidator,
    ObsModuleConfig,
    QuaboUidsValidator,
)

PID_FILE = "tmp/interleave.pid"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("panoseti.interleave")


class InterleaveController:
    """Orchestrates rapid switching between observing modes across all Quabos.
    
    This controller manages a cyclical observing schedule defined in the 
    data configuration. It ensures that hardware is correctly reconfigured 
    (MAROC and FPGA registers) at every state transition.
    """
    MAX_THREADS = 8
    def __init__(self, data_config: DataConfigValidator | dict[str, Any], obs_config: ObsConfigValidator | dict[str, Any],
                 daq_config: DaqConfigValidator | dict[str, Any], quabo_uids: QuaboUidsValidator | dict[str, Any],
                 quabo_info: dict[str, Any], network_config: NetworkConfigValidator | dict[str, Any],
                 dry_run: bool = False, max_cycles: int | None = None) -> None:
        """Initialize the Interleave Controller and verify system state.

        Args:
            data_config: Acquisition configuration including interleave states.
            obs_config: Physical observatory configuration.
            daq_config: DAQ node networking configuration.
            quabo_uids: Quabo hardware UID registry.
            quabo_info: Detailed Quabo metadata.
            network_config: Network routing/port forwarding rules.
            dry_run: If True, simulate hardware commands without sending them.
            max_cycles: Optional limit on the number of observing cycles to run.
        """

        if isinstance(data_config, dict):
            data_config = DataConfigValidator(**data_config)
        if isinstance(obs_config, dict):
            obs_config = ObsConfigValidator(**obs_config)
        if isinstance(daq_config, dict):
            daq_config = DaqConfigValidator(**daq_config)
        if isinstance(quabo_uids, dict):
            quabo_uids = QuaboUidsValidator(**quabo_uids)
        if isinstance(network_config, dict):
            network_config = NetworkConfigValidator(**network_config)

        self.keep_running = True
        self.dry_run = dry_run
        self.max_cycles = max_cycles
        self.stats: dict[str, Any] = {
            "total_cycles": 0,
            "total_switch_overhead_sec": 0.0,
            "overhead": []
        }

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

        self.data_config = data_config
        self.interleave_cfg = data_config.interleave or InterleaveConfig()
        self.obs_config = obs_config
        self.daq_config = daq_config
        self.quabo_uids = quabo_uids
        self.quabo_info = quabo_info
        self.network_config = network_config

        self.modules = config_file.get_modules(obs_config)
        self.quabos: dict[int, list[quabo_driver.QUABO]] = {}  # map module_id to quabo_driver instances
        for module in self.modules:
            base_ip_addr = str(module.ip_addr)
            module_id = module.id if module.id is not None else -1
            self.quabos[module_id] = []
            for i in range(4):
                uid = util.quabo_uid(module.model_dump(), quabo_uids.model_dump(), i)
                if not uid:
                    continue
                ip_ports = util.get_quabo_ip_port(base_ip_addr, i, network_config)
                real_ip = ip_ports['ip_addr']
                cmd_port = ip_ports['cmd_port']
                q = quabo_driver.QUABO(real_ip, cmd_port)
                self.quabos[module_id].append(q) # use a list to retain sequential ordering


        # self.executor = ThreadPoolExecutor(max_workers=self.MAX_THREADS)

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)

        if self.dry_run:
            logger.info("=== DRY RUN MODE ENABLED: Hardware commands will be simulated ===")

    def _acquire_lock(self) -> None:
        """Ensure only one instance of the interleaver is active using a PID file.
        
        Raises:
            SystemExit: If another instance is already running.
        """
        if os.path.exists(PID_FILE):
            try:
                with open(PID_FILE) as f:
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

    def _release_lock(self) -> None:
        """Remove the interleaver PID file."""
        if os.path.exists(PID_FILE):
            with contextlib.suppress(OSError):
                os.remove(PID_FILE)

    def _handle_shutdown_signal(self, signum: int, frame: Any) -> None:
        """Gracefully break the observing loop on SIGINT/SIGTERM."""
        if self.keep_running:
            logger.warning("Shutdown signal received. Breaking cycle to restore defaults...")
            self.keep_running = False

    def _broadcast_acq_mode(self, daq_params: quabo_driver.DAQ_PARAMS) -> None:
        """Broadcast acquisition mode parameters to all Quabos.
        
        Broadcasts are performed sequentially across modules.

        Args:
            daq_params: The Quabo-level DAQ parameters to send.
        """
        if self.dry_run:
            #logger.info(f"[DRY-RUN] Simulating ACQ broadcast: do_image={daq_params.do_image}, do_ph={daq_params.do_ph}")
            return

        def send_acq_mode_to_module(module_id: int) -> None:
            """Sequentially broadcast ACQ mode to Quabos in the Q0, Q1, Q2, Q3 order."""
            quabos = self.quabos[module_id]
            for q in quabos:
                q.send_daq_params(daq_params)
        #futures = []
        for mid in self.quabos:
            #futures.append(self.executor.submit(send_acq_mode_to_module, mid))
            send_acq_mode_to_module(mid)
        #for f in as_completed(futures):
        #    f.result()

    def _reconfigure_quabos(self, next_state_data_config: DataConfigValidator) -> None:
        """Reconfigure MAROC and FPGA registers for a specific observing mode.

        Args:
            next_state_data_config: The configuration model for the target mode.
        """
        if self.dry_run:
            return

        def reconfig_modules(modules: list[ObsModuleConfig]) -> None:
            pano_config.do_maroc_config(
                [m.model_dump() for m in modules], self.quabo_uids.model_dump(), self.quabo_info,
                next_state_data_config.model_dump(), self.obs_config.model_dump(), self.daq_config.model_dump(),
                self.network_config, 
                verbose=False, write_config=False, do_log=False
            )
            pano_config.do_mask_config(
                [m.model_dump() for m in modules], next_state_data_config.model_dump(), 
                self.network_config,
                self.quabo_uids.model_dump(), verbose=False, write_config=False, do_flush_rx_buf=False, do_log=False
            )

        reconfig_modules(self.modules)
        #futures = [self.executor.submit(reconfig_module, module) for module in self.modules]
        #for f in as_completed(futures): f.result()

    def generate_state_config(self, movie_key: str | None, ph_key: str | None) -> DataConfigValidator:
        """Generate a mode-specific data configuration for a target interleave state.
        
        Merges mode overrides into the base configuration.

        Args:
            movie_key: Key in model_extra or 'image' to use for movie mode.
            ph_key: Key in model_extra or 'pulse_height' to use for PH mode.

        Returns:
            A new DataConfigValidator reflecting the target observing mode.
        """
        temp_dict = self.data_config.model_dump()
        temp_dict.pop('image', None)
        temp_dict.pop('pulse_height', None)
        
        # Access extra modes via model_extra if not root 'image'/'pulse_height'
        extra = self.data_config.model_extra or {}
        
        if movie_key:
            if movie_key == 'image':
                temp_dict['image'] = self.data_config.image.model_dump() if self.data_config.image else None
            else:
                temp_dict['image'] = extra.get(movie_key)
        if ph_key:
            if ph_key == 'pulse_height':
                temp_dict['pulse_height'] = self.data_config.pulse_height.model_dump() if self.data_config.pulse_height else None
            else:
                temp_dict['pulse_height'] = extra.get(ph_key)
        
        return DataConfigValidator(**temp_dict)

    def _sleep_until(self, target_time: float, spin_wait_threshold: float = 0.005) -> None:
        """High-precision sleep that uses hybrid sleep/spin logic.

        Args:
            target_time: Perf counter value to sleep until.
            spin_wait_threshold: Threshold below which to spin instead of sleep.
        """
        while self.keep_running:
            now = time.perf_counter()
            remaining = target_time - now
            if remaining <= spin_wait_threshold:
                break
            time.sleep(remaining - spin_wait_threshold)


    def run_loop(self) -> None:
        """Execute the main interleaving observing loop until stopped."""
        # Must import start dynamically to avoid circular import errors
        from start import get_daq_params

        if not self.interleave_cfg.enable:
            #logger.info("Interleaving disabled in config. Exiting.")
            self._release_lock()
            return

        states = self.interleave_cfg.states
        if not states:
            self._release_lock()
            return

        stop_daq_params = quabo_driver.DAQ_PARAMS(False, 0, False, False, False)

        try:
            while self.keep_running:
                if self.max_cycles and self.stats["total_cycles"] >= self.max_cycles:
                    logger.info(f"Max cycles ({self.max_cycles}) reached. Ending run_loop.")
                    break

                for state in states:
                    name = state.state_name
                    duration = state.duration_seconds

                    # make the next interleave state appear as the "default" state
                    # this lets us use existing production helper functions to config
                    next_state_data_config = self.generate_state_config(
                        state.movie_mode_config,
                        state.pulse_height_mode_config
                    )
                    start_daq_params = get_daq_params(next_state_data_config)

                    t_overhead_start = time.perf_counter()

                    # 1. Stop data flow
                    self._broadcast_acq_mode(stop_daq_params)
                    # 2. Reconfigure Quabos
                    self._reconfigure_quabos(next_state_data_config)
                    # 3. Start data flow
                    self._broadcast_acq_mode(start_daq_params)

                    overhead = time.perf_counter() - t_overhead_start
                    self.stats["total_switch_overhead_sec"] += overhead
                    self.stats["overhead"].append(overhead)

                    logger.info(
                        f"\n--- Entering State: {name} (Duration: {duration}s) ---"
                        f"Hardware configured in {overhead * 1e3:.4f} ms. Actively observing for {duration}s..."
                    )

                    # Relative Active Scheduling: Guarantee the full duration occurs AFTER configuration
                    self._sleep_until(time.perf_counter() + duration)

                # Move this inside if needed, or remove the 'break' that confuses mypy
                self.stats["total_cycles"] += 1

        except Exception as e:
            logger.error(f"Error in interleaving loop: {e}", exc_info=True)
        finally:
            start_default_daq_params = get_daq_params(self.data_config)
            self._teardown(stop_daq_params, start_default_daq_params)

    def _teardown(self, stop_daq_params: quabo_driver.DAQ_PARAMS, start_default_daq_params: quabo_driver.DAQ_PARAMS) -> None:
        """Restores Quabos to default settings and cleans up."""
        overhead_list = self.stats['overhead']
        if overhead_list:
            logger.critical(fr"""Overhead stats: 
    count:	{len(overhead_list)}
    mean:	{np.mean(overhead_list) * 1e3:.5f} ms
    stdev:	{np.std(overhead_list) * 1e3:.5f} ms
    median:	{np.median(overhead_list) * 1e3:.5f} ms
    min:	{np.min(overhead_list) * 1e3:.5f} ms
    max:	{np.max(overhead_list) * 1e3:.5f} ms""")


        if self.dry_run:
            #logger.info("[DRY-RUN] Teardown initiated. Simulating hardware default restoration.")
            self._release_lock()
            return

        logger.info("Teardown initiated. Restoring default hardware configuration...")
        # stop_daq_params = quabo_driver.DAQ_PARAMS(False, 0, False, False, False)
        try:
            # from start import get_daq_params
            # get default daq params
            # next_state_data_config = self.data_config

            # Forcefully drop old tasks and spin up a fresh pool to guarantee teardown commands aren't stuck behind deadlocked threads.
            # self.executor.shutdown(wait=False)
            # self.executor = ThreadPoolExecutor(max_workers=self.MAX_THREADS)

            # restore default parameters
            self._broadcast_acq_mode(stop_daq_params)
            self._reconfigure_quabos(self.data_config)
            self._broadcast_acq_mode(start_default_daq_params)

            logger.info("Hardware defaults restored successfully.")
        except Exception as e:
            logger.exception(f"Failed to restore hardware defaults: {e}")
        finally:
            # self.executor.shutdown(wait=False)
            for quabos in self.quabos.values():
                for q in quabos:
                    q.close()
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
