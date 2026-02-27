#!/usr/bin/env python3
"""
PANOSETI Interleaved Observation Controller

This script rapidly switches Quabo FPGA and MAROC registers between different
observing modes (e.g., Pulse Height mode and Image mode) to achieve interleaved
astrometry and trigger-based science data collection.

Usage:
    python tools/interleave.py --help
    python tools/interleave.py --config configs/data_config.json --verbose
    python tools/interleave.py --dry-run
"""

import time
import logging
import sys
import os
import argparse
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure the parent control directory is in the path for module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from driver import quabo_driver
import config as pano_config
from utils import config_file
from utils import util

# Import the Pydantic models for fail-fast validation
from tools.interleave_helper.pydantic_config_models import DataConfigValidator

# Try to use rich for pretty logging, fallback to standard if unavailable
try:
    from rich.logging import RichHandler
    from rich.console import Console

    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None

logger = logging.getLogger("panoseti.interleave")


def sleep_until(target_time: float, spin_wait_threshold: float = 0.005) -> None:
    """
    High-precision sleep until an absolute target time (based on time.perf_counter).

    Uses OS sleep for the bulk of the wait to save CPU, then busy-waits
    the final few milliseconds for sub-millisecond precision.
    """
    # Bulk sleep (give back to OS)
    while True:
        now = time.perf_counter()
        remaining = target_time - now
        if remaining <= spin_wait_threshold:
            break
        # Sleep slightly less than remaining to ensure we don't oversleep
        time.sleep(remaining - spin_wait_threshold)

    # Spin lock for the last few milliseconds (high CPU, high precision)
    while time.perf_counter() < target_time:
        pass

class InterleaveController:
    """
    Manages the timed loop of reconfiguring Quabos for interleaved observations.

    This class validates the requested data config, establishes connections to
    all active Quabos, and uses a ThreadPool to broadcast state changes in parallel
    to minimize dead time between observing states.
    """

    def __init__(self, data_config: Dict[str, Any], obs_config: Dict[str, Any],
                 daq_config: Dict[str, Any], quabo_uids: Dict[str, Any],
                 quabo_info: List[Dict[str, Any]], network_config: Dict[str, Any],
                 dry_run: bool = False):
        """
        Initialize the InterleaveController and validate configurations.

        Args:
            data_config: The raw dictionary loaded from data_config.json.
            obs_config: The observatory hardware configuration.
            daq_config: The DAQ node mapping configuration.
            quabo_uids: The mapping of module IDs to Quabo UIDs.
            quabo_info: The list of Quabo hardware details.
            network_config: Port forwarding and IP configurations.
            dry_run: If True, simulate network commands without executing them.
        """
        self.dry_run = dry_run
        self.stats = {"total_cycles": 0, "total_switch_overhead_sec": 0.0, "total_observe_sec": 0.0}

        # 1. Internal Validation: Ensure data_config is valid before we proceed
        logger.info("Validating data configuration schema...")
        try:
            validated = DataConfigValidator(**data_config)
            self.data_config = data_config
            self.interleave_cfg = data_config.get("interleave", {})
            logger.info("Configuration validated successfully.")
        except Exception as e:
            logger.error(f"Data configuration validation failed: {e}")
            raise

        self.obs_config = obs_config
        self.daq_config = daq_config
        self.quabo_uids = quabo_uids
        self.quabo_info = quabo_info
        self.network_config = network_config

        # 2. Extract modules utilizing existing `config_file.py` patterns
        self.modules = config_file.get_modules(obs_config)
        logger.info(f"Found {len(self.modules)} modules in observatory config.")

        # 3. Instantiate Quabo drivers
        self.quabos: List[quabo_driver.QUABO] = []
        for module in self.modules:
            for i in range(4):
                uid = util.quabo_uid(module, quabo_uids, i)
                if not uid:
                    continue

                # Use standard util pattern to map IPs and Ports
                ip_ports = util.get_quabo_ip_port(module['ip_addr'], i, network_config)
                real_ip = ip_ports['ip_addr']
                cmd_port = ip_ports['cmd_port']

                if not self.dry_run:
                    self.quabos.append(quabo_driver.QUABO(real_ip, cmd_port))
                else:
                    logger.debug(f"[Dry Run] Discovered Quabo at {real_ip}:{cmd_port} (UID: {uid})")

        # Create the threadpool for maximum parallel network performance
        self.executor = ThreadPoolExecutor(max_workers=len(self.modules) + 4)

    def _broadcast_acq_mode(self, daq_params: quabo_driver.DAQ_PARAMS) -> None:
        """
        Send DAQ parameters in parallel to all Quabos to start/stop packet flow.

        Args:
            daq_params: A configured quabo_driver.DAQ_PARAMS object.
        """
        if self.dry_run:
            logger.info(f"[Dry Run] Broadcasting ACQ Mode: Image={daq_params.do_image}, PH={daq_params.do_ph}")
            # time.sleep(0.01)  # Simulate slight network latency
            return

        def send_acq(q: quabo_driver.QUABO):
            q.send_daq_params(daq_params)

        futures = [self.executor.submit(send_acq, q) for q in self.quabos]
        for f in as_completed(futures):
            f.result()  # Raises exception if network call failed

    def _reconfigure_quabos(self, state_config_dict: Dict[str, Any]) -> None:
        """
        Reconfigures all quabos based on a synthesized data_config dict for the current state.
        Distributes `do_maroc_config` across modules concurrently.

        Args:
            state_config_dict: A mocked data_config dict representing the target state.
        """
        if self.dry_run:
            logger.info("[Dry Run] Reconfiguring MAROC/FPGA registers...")
            # time.sleep(0.5)  # Simulate the time it takes to flash registers
            return

        def reconfig_module(module):
            pano_config.do_maroc_config(
                [module], self.quabo_uids, self.quabo_info,
                state_config_dict, self.obs_config, self.daq_config,
                self.network_config, verbose=False
            )

        futures = [self.executor.submit(reconfig_module, module) for module in self.modules]
        for f in as_completed(futures):
            f.result()

    def generate_state_dict(self, state_def: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a mock data_config dictionary for the existing configuration system.
        Replaces the root 'image' and 'pulse_height' keys with suffixed versions.
        """
        temp_dict = self.data_config.copy()
        temp_dict.pop('image', None)
        temp_dict.pop('pulse_height', None)

        if state_def.get("movie_mode_config"):
            temp_dict['image'] = self.data_config[state_def["movie_mode_config"]]
        if state_def.get("pulse_height_mode_config"):
            temp_dict['pulse_height'] = self.data_config[state_def["pulse_height_mode_config"]]

        return temp_dict

    def build_daq_params(self, state_dict: Dict[str, Any]) -> quabo_driver.DAQ_PARAMS:
        """
        Construct a quabo_driver.DAQ_PARAMS object based on the state data config.
        """
        do_img = 'image' in state_dict
        do_ph = 'pulse_height' in state_dict

        image_us = 0
        image_8bit = False

        if do_img:
            image_us = state_dict['image'].get('integration_time_usec', 0)
            if state_dict['image'].get('quabo_sample_size', 0) == 8:
                image_8bit = True

        bl_subtract = True
        do_any_trigger = False
        do_group_ph_frames = False

        if do_ph:
            ph_cfg = state_dict['pulse_height']
            if 'any_trigger' in ph_cfg:
                do_any_trigger = True
                do_group_ph_frames = bool(ph_cfg['any_trigger'].get('group_ph_frames', 0))

        return quabo_driver.DAQ_PARAMS(
            do_image=do_img,
            image_us=image_us,
            image_8bit=image_8bit,
            do_ph=do_ph,
            bl_subtract=bl_subtract,
            do_any_trigger=do_any_trigger,
            do_group_ph_frames=do_group_ph_frames
        )

    def get_stop_daq_params(self) -> quabo_driver.DAQ_PARAMS:
        """Returns standard params to gracefully stop DAQ flow."""
        return quabo_driver.DAQ_PARAMS(
            do_image=False, image_us=0, image_8bit=False,
            do_ph=False, bl_subtract=True
        )

    def print_profiling_stats(self):
        """Output tracking stats regarding timing efficiency."""
        logger.info("\n=== Interleaving Profiling Report ===")
        logger.info(f"Total Cycles Completed: {self.stats['total_cycles']}")
        if self.stats['total_cycles'] > 0:
            avg_overhead = self.stats['total_switch_overhead_sec'] / self.stats['total_cycles']
            logger.info(f"Average Switching Overhead: {avg_overhead:.3f} seconds/switch")
            logger.info(f"Total Observation Time: {self.stats['total_observe_sec']:.2f} seconds")

    def run_loop(self) -> None:
        """
        Executes the main infinite observing loop using absolute time scheduling.
        """
        if not self.interleave_cfg.get("enable", False):
            logger.info("Interleaving is disabled. Exiting.")
            return

        states = self.interleave_cfg.get("states", [])
        stop_params = self.get_stop_daq_params()

        # Get the absolute starting time for the entire schedule
        schedule_start_time = time.perf_counter()
        next_state_time = schedule_start_time

        try:
            while True:
                for state in states:
                    name = state["state_name"]
                    duration = state["duration_seconds"]

                    # Calculate EXACTLY when this state should end
                    next_state_time += duration

                    logger.info(f"\n--- Entering State: {name} ---")
                    t_overhead_start = time.perf_counter()

                    # 1. STOP DAQ FLOW
                    self._broadcast_acq_mode(stop_params)

                    # 2. RECONFIGURE
                    state_dict = self.generate_state_dict(state)
                    self._reconfigure_quabos(state_dict)

                    # 3. START DAQ FLOW
                    daq_params = self.build_daq_params(state_dict)
                    self._broadcast_acq_mode(daq_params)

                    # Track the overhead of the switching logic
                    t_overhead_end = time.perf_counter()
                    overhead = t_overhead_end - t_overhead_start
                    self.stats["total_switch_overhead_sec"] += overhead
                    logger.debug(f"Hardware switch overhead: {overhead:.5f}s")

                    # If our switch took LONGER than the intended state duration, we are falling behind.
                    if time.perf_counter() > next_state_time:
                        logger.warning(
                            f"Warning: Switching overhead ({overhead:.3f}s) exceeded state duration ({duration}s). Schedule is slipping!")
                        # Reset the anchor to prevent infinite catch-up loops
                        next_state_time = time.perf_counter()
                        continue

                    # 4. OBSERVE (Precision wait until the absolute end time)
                    logger.info(f"Observing... (Scheduled to end in {next_state_time - time.perf_counter():.3f}s)")

                    sleep_until(next_state_time)

                    self.stats["total_observe_sec"] += duration
                    self.stats["total_cycles"] += 1

        except KeyboardInterrupt:
            logger.warning("\nInterleaving aborted by user. Stopping data flow gracefully.")
            self._broadcast_acq_mode(stop_params)
        finally:
            self.print_profiling_stats()
            self.executor.shutdown(wait=False)
            if not self.dry_run:
                for q in self.quabos:
                    q.close()


def main():
    parser = argparse.ArgumentParser(description="PANOSETI Interleaving Observation Controller")
    # parser.add_argument("--config", type=str, default="",
    #                     help="Path to the data_config.json file (default: configs/data_config.json)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Parse and validate the configuration file, then exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the timing logic and print commands without sending UDP packets.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG level logging for detailed trace output.")

    args = parser.parse_args()

    # Configure Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    handlers = [RichHandler(rich_tracebacks=True)] if RICH_AVAILABLE else [logging.StreamHandler()]
    logging.basicConfig(level=log_level, format="%(message)s", datefmt="[%X]", handlers=handlers)

    try:
        # Load all base configurations
        logger.info("Loading base observatory configurations...")
        obs_config = config_file.get_obs_config()
        daq_config = config_file.get_daq_config()
        quabo_uids = config_file.get_quabo_uids()
        quabo_info = config_file.get_quabo_info()
        network_config = config_file.get_network_config()

        # Load data config using the standard util (we validate it inside the class)
        data_config = config_file.get_data_config()

        if args.validate_only:
            # Re-run validation explicitly to trigger the rich printout
            DataConfigValidator(**data_config)
            logger.info("Validation complete. Exiting.")
            sys.exit(0)

        # Initialize and run
        controller = InterleaveController(
            data_config=data_config,
            obs_config=obs_config,
            daq_config=daq_config,
            quabo_uids=quabo_uids,
            quabo_info=quabo_info,
            network_config=network_config,
            dry_run=args.dry_run
        )
        controller.run_loop()

    except Exception as e:
        logger.error(f"Execution Failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()