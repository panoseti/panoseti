#!/usr/bin/env python3
"""
interleave.py

PANOSETI Interleaved Observation Controller.
Runs as a background daemon during an active observation to rapidly
switch Quabo FPGA and MAROC registers between different observing modes.

Features:
- Precise NTP-timestamped event logging for post-observation analysis.
- Robust state management with deepcopy to ensure pristine teardown.
- Graceful shutdown handling via system signals.
"""

import time
import logging
import argparse
import sys
import os
import signal
import copy
import csv
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from driver import quabo_driver
import config as pano_config
from utils import config_file, util

PID_FILE = "tmp/interleave.pid"
EVENT_LOG_FILE = "logs/interleave_events.csv"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("panoseti.interleave")


class InterleaveController:
    """
    Manages the lifecycle, hardware dispatching, and logging of interleaved
    observing modes for PANOSETI Quabo modules.
    """

    def __init__(self, data_config: Dict[str, Any], obs_config: Dict[str, Any],
                 daq_config: Dict[str, Any], quabo_uids: Dict[str, Any],
                 quabo_info: List[Dict[str, Any]], network_config: Dict[str, Any],
                 dry_run: bool = False, max_cycles: Optional[int] = None):

        self.keep_running = True
        self.dry_run = dry_run
        self.max_cycles = max_cycles
        self.stats = {"total_cycles": 0, "total_switch_overhead_sec": 0.0}

        self._acquire_lock()

        # Freeze a pristine copy of the initial data config to guarantee safe teardowns
        self.original_data_config = copy.deepcopy(data_config)
        self.interleave_cfg = self.original_data_config.get("interleave", {})

        self.obs_config = obs_config
        self.daq_config = daq_config
        self.quabo_uids = quabo_uids
        self.quabo_info = quabo_info
        self.network_config = network_config

        self.modules = config_file.get_modules(obs_config)
        self.quabos: List[quabo_driver.QUABO] = []
        for module in self.modules:
            for i in range(4):
                uid = util.quabo_uid(module, quabo_uids, i)
                if not uid: continue
                ip_ports = util.get_quabo_ip_port(module['ip_addr'], i, network_config)
                self.quabos.append(quabo_driver.QUABO(ip_ports['ip_addr'], ip_ports['cmd_port']))

        self.executor = ThreadPoolExecutor(max_workers=len(self.modules) + 4)

        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)

        self.csv_file = None
        self.csv_writer = None
        self._init_event_logger()

        if self.dry_run:
            logger.info("=== DRY RUN MODE ENABLED: Hardware commands will be simulated ===")

    def _acquire_lock(self) -> None:
        """Ensures at most one instance of interleave.py is running via PID file."""
        if os.path.exists(PID_FILE):
            with open(PID_FILE, "r") as f:
                try:
                    old_pid = int(f.read().strip())
                    os.kill(old_pid, 0)
                    logger.critical(
                        f"CRITICAL: Another interleave process (PID {old_pid}) is running.\n"
                        "Run `python config.py --stop-interleave` to resolve."
                    )
                    sys.exit(1)
                except (ValueError, OSError):
                    pass  # Stale PID, overwrite

        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))

    def _release_lock(self) -> None:
        """Removes the PID file upon exit."""
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)

    def _handle_shutdown_signal(self, signum: int, frame: Any) -> None:
        """Catches SIGTERM/SIGINT to gracefully break the loop and trigger teardown."""
        logger.warning("Shutdown signal received. Preparing to restore original defaults...")
        self.keep_running = False

    def _init_event_logger(self) -> None:
        """Initializes the CSV file for highly accurate timestamp logging."""
        os.makedirs(os.path.dirname(EVENT_LOG_FILE), exist_ok=True)
        file_exists = os.path.exists(EVENT_LOG_FILE) and os.path.getsize(EVENT_LOG_FILE) > 0

        self.csv_file = open(EVENT_LOG_FILE, mode='a', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        if not file_exists:
            self.csv_writer.writerow(["unix_timestamp", "utc_datetime", "event_type", "state_name", "details"])

    def _log_event(self, event_type: str, state_name: str, details: str = "") -> None:
        """
        Records an event with an absolute NTP-synchronized timestamp.
        Immediately flushes to disk to prevent data loss on crash.
        """
        now_ts = time.time()
        now_utc = datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat()

        # Log to terminal (optional, keeping clean)
        logger.debug(f"Event: {event_type} | State: {state_name} | {details}")

        if self.csv_writer and self.csv_file:
            self.csv_writer.writerow([f"{now_ts:.6f}", now_utc, event_type, state_name, details])
            self.csv_file.flush()
            os.fsync(self.csv_file.fileno())  # Guarantee write to disk

    def _broadcast_acq_mode(self, daq_params: quabo_driver.DAQ_PARAMS) -> None:
        """Broadcasts acquisition mode to all Quabos concurrently."""
        if self.dry_run:
            logger.info(f"[DRY-RUN] Simulating ACQ broadcast: img={daq_params.do_image}, ph={daq_params.do_ph}")
            return

        def send_acq(q: quabo_driver.QUABO):
            q.send_daq_params(daq_params)

        futures = [self.executor.submit(send_acq, q) for q in self.quabos]
        for f in as_completed(futures):
            f.result()

    def _reconfigure_quabos(self, state_config_dict: Dict[str, Any]) -> None:
        """Calls the main config scripts to rewrite MAROC registers AND FPGA Trigger Masks."""
        if self.dry_run:
            modes = [k for k in state_config_dict.keys() if k in ['image', 'pulse_height']]
            logger.info(f"[DRY-RUN] Simulating MAROC and MASK reconfig. Active modes: {modes}")
            return

        def reconfig_module(module):
            # 1. Reconfigure MAROC DACs and thresholds (Serial)
            pano_config.do_maroc_config(
                [module], self.quabo_uids, self.quabo_info,
                state_config_dict, self.obs_config, self.daq_config,
                self.network_config, verbose=False
            )

            # 2. Reconfigure FPGA Trigger Masks (CRITICAL for resetting frame rates)
            pano_config.do_mask_config(
                [module], state_config_dict, self.network_config,
                self.quabo_uids, verbose=False
            )

        futures = [self.executor.submit(reconfig_module, module) for module in self.modules]
        for f in as_completed(futures):
            f.result()

    def generate_state_dict(self, movie_key: Optional[str], ph_key: Optional[str]) -> Dict[str, Any]:
        """
        Generates a configuration dictionary strictly matching the structure expected
        by do_maroc_config, built freshly from the pristine original_data_config.
        """
        temp_dict = copy.deepcopy(self.original_data_config)
        # Wipe base modes
        temp_dict.pop('image', None)
        temp_dict.pop('pulse_height', None)

        # Insert target modes as the new base modes
        if movie_key and movie_key in self.original_data_config:
            temp_dict['image'] = copy.deepcopy(self.original_data_config[movie_key])
        if ph_key and ph_key in self.original_data_config:
            temp_dict['pulse_height'] = copy.deepcopy(self.original_data_config[ph_key])

        return temp_dict

    def build_daq_params(self, state_dict: Dict[str, Any]) -> quabo_driver.DAQ_PARAMS:
        """Parses the current state config dict into hardware DAQ_PARAMS."""
        do_img = 'image' in state_dict
        do_ph = 'pulse_height' in state_dict

        image_us = state_dict['image'].get('integration_time_usec', 0) if do_img else 0
        image_8bit = (state_dict['image'].get('quabo_sample_size', 0) == 8) if do_img else False

        do_any_trigger = False
        do_group_ph_frames = False
        if do_ph and 'any_trigger' in state_dict['pulse_height']:
            do_any_trigger = True
            do_group_ph_frames = bool(state_dict['pulse_height']['any_trigger'].get('group_ph_frames', 0))

        return quabo_driver.DAQ_PARAMS(
            do_image=do_img, image_us=image_us, image_8bit=image_8bit,
            do_ph=do_ph, bl_subtract=True, do_any_trigger=do_any_trigger,
            do_group_ph_frames=do_group_ph_frames
        )

    def _sleep_until(self, target_time: float, spin_wait_threshold: float = 0.005) -> None:
        """High-precision hybrid sleep. Yields to OS, then spins for the last few ms."""
        while self.keep_running:
            now = time.perf_counter()
            remaining = target_time - now
            if remaining <= spin_wait_threshold:
                break
            time.sleep(remaining - spin_wait_threshold)

        while self.keep_running and time.perf_counter() < target_time:
            pass

    def run_loop(self) -> None:
        """Main operational loop executing the interleave schedule."""
        if not self.interleave_cfg.get("enable", False):
            logger.info("Interleaving disabled in config. Exiting.")
            self._release_lock()
            return

        states = self.interleave_cfg.get("states", [])
        stop_params = quabo_driver.DAQ_PARAMS(False, 0, False, False, True)

        schedule_start_time = time.perf_counter()
        next_state_time = schedule_start_time

        self._log_event("INTERLEAVE_START", "GLOBAL", "Starting interleave daemon")

        try:
            while self.keep_running:
                if self.max_cycles and self.stats["total_cycles"] >= self.max_cycles:
                    logger.info(f"Max cycles ({self.max_cycles}) reached. Ending run_loop.")
                    break

                for state in states:
                    if not self.keep_running:
                        break

                    name = state["state_name"]
                    duration = state["duration_seconds"]
                    next_state_time += duration

                    logger.info(f"\n--- Entering State: {name} (Duration: {duration}s) ---")
                    self._log_event("SWITCH_START", name, "Stopping DAQ and reconfiguring")

                    t_overhead_start = time.perf_counter()

                    self._broadcast_acq_mode(stop_params)
                    state_dict = self.generate_state_dict(
                        state.get("movie_mode_config"),
                        state.get("pulse_height_mode_config")
                    )
                    self._reconfigure_quabos(state_dict)
                    self._broadcast_acq_mode(self.build_daq_params(state_dict))

                    overhead = time.perf_counter() - t_overhead_start
                    self.stats["total_switch_overhead_sec"] += overhead

                    self._log_event("OBSERVE_START", name, f"Reconfigured in {overhead:.3f}s")

                    if time.perf_counter() > next_state_time:
                        logger.warning("Switch overhead exceeded state duration. Resetting timeline.")
                        next_state_time = time.perf_counter()
                        continue

                    self._sleep_until(next_state_time)
                    self._log_event("OBSERVE_END", name, "Observation duration complete")

                self.stats["total_cycles"] += 1

        except Exception as e:
            logger.error(f"Error in interleaving loop: {e}", exc_info=True)
            self._log_event("ERROR", "GLOBAL", str(e))
        finally:
            self._teardown(stop_params)

    def _teardown(self, stop_params: quabo_driver.DAQ_PARAMS) -> None:
        """
        Safely halts operations and forcefully restores the Quabos to the exact
        state defined by the original, pristine initial configuration file.
        """
        logger.info("Teardown initiated. Restoring pure hardware defaults...")
        self._log_event("TEARDOWN_START", "GLOBAL", "Restoring pristine original configuration")
        try:
            if not self.dry_run:
                self._broadcast_acq_mode(stop_params)

                # We do NOT generate from current state; we generate strictly from the original root config.
                default_dict = self.generate_state_dict(
                    "image" if "image" in self.original_data_config else None,
                    "pulse_height" if "pulse_height" in self.original_data_config else None
                )

                self._reconfigure_quabos(default_dict)
                self._broadcast_acq_mode(self.build_daq_params(default_dict))

            logger.info("Hardware defaults restored successfully.")
            self._log_event("TEARDOWN_COMPLETE", "GLOBAL", "Hardware defaults restored")

        except Exception as e:
            logger.error(f"Failed to restore hardware defaults: {e}")
            self._log_event("TEARDOWN_ERROR", "GLOBAL", str(e))
        finally:
            self.executor.shutdown(wait=False)
            if not self.dry_run:
                for q in self.quabos:
                    q.close()

            if self.csv_file:
                self.csv_file.close()

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
        if controller:
            controller._release_lock()
        sys.exit(1)