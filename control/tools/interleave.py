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
import sys
import os
import copy
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from driver import quabo_driver
import config as pano_config
from utils import config_file, util

PID_FILE = "tmp/interleave.pid"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("panoseti.interleave")


class InterleaveController:
    def __init__(self, data_config: Dict[str, Any], obs_config: Dict[str, Any],
                 daq_config: Dict[str, Any], quabo_uids: Dict[str, Any],
                 quabo_info: List[Dict[str, Any]], network_config: Dict[str, Any],
                 dry_run: bool = False, max_cycles: Optional[int] = None):

        self.keep_running = True
        self.dry_run = dry_run
        self.max_cycles = max_cycles
        self.stats = {"total_cycles": 0, "total_switch_overhead_sec": 0.0}

        self.data_config = data_config
        self.obs_config = obs_config
        self.daq_config = daq_config
        self.quabo_uids = quabo_uids
        self.quabo_info = quabo_info
        self.network_config = network_config

        # Extract modules from obs_config
        self.modules = []
        for dome in self.obs_config.get('domes', []):
            for module in dome.get('modules', []):
                self.modules.append(module)

        self.interleave_config = self.data_config.get('interleave', {})
        self.states = self.interleave_config.get('states', [])

        # We will use ThreadPoolExecutor for broadcasting DAQ acq mode commands across nodes
        self.executor = ThreadPoolExecutor(max_workers=len(self.modules) * 4 if self.modules else 1)

        # Setup base quabo connections for broadcasting DAQ start/stop commands
        self.quabos = []
        for module in self.modules:
            for i in range(4):
                ip_ports = util.get_quabo_ip_port(module['ip_addr'], i, self.network_config)
                self.quabos.append(quabo_driver.QUABO(ip_ports['ip_addr']))

    def _broadcast_acq_mode(self, daq_params):
        """Broadcasts DAQ parameters to all Quabos concurrently to start/stop data flow."""
        if self.dry_run:
            logger.info(f"DRY RUN: Broadcasting DAQ params {daq_params.__dict__}")
            return

        def send_to_quabo(q):
            try:
                original_timeout = q.sock.gettimeout()
                q.sock.settimeout(0.05)
                q.send_daq_params(daq_params)
                q.sock.settimeout(original_timeout)
            except socket.timeout:
                pass  # UDP drops are ignored per protocol
            except Exception as e:
                logger.debug(f"Error sending DAQ params to {q.ip}: {e}")

        futures = [self.executor.submit(send_to_quabo, q) for q in self.quabos]
        for f in as_completed(futures):
            pass

    def build_daq_params(self, active_data_config: Dict[str, Any]) -> quabo_driver.DAQ_PARAMS:
        """Constructs DAQ_PARAMS object based on the current data configuration."""
        do_image = 'image' in active_data_config
        do_image_8bit = 'image_8bit' in active_data_config
        do_ph = 'pulse_height' in active_data_config

        image_us = 0
        if do_image:
            image_us = active_data_config['image'].get('integration_time_usec', 0)
        elif do_image_8bit:
            image_us = active_data_config['image_8bit'].get('integration_time_usec', 0)

        return quabo_driver.DAQ_PARAMS(
            do_image=do_image,
            image_us=image_us,
            image_8bit=do_image_8bit,
            do_ph=do_ph,
            bl_subtract=False
        )

    def prepare_state_config(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Creates a patched copy of data_config that maps the requested interleave mode to default keys."""
        state_config = copy.deepcopy(self.data_config)

        # Remove default keys to prevent overlapping configs
        for key in ['image', 'image_8bit', 'pulse_height']:
            state_config.pop(key, None)

        # Map movie mode config
        movie_mode = state.get('movie_mode_config')
        if movie_mode and movie_mode in self.data_config:
            # Simple heuristic: if '8bit' is in the key name, map to 'image_8bit', else 'image'
            target_key = 'image_8bit' if '8bit' in movie_mode.lower() else 'image'
            state_config[target_key] = self.data_config[movie_mode]

        # Map pulse height mode config
        ph_mode = state.get('pulse_height_mode_config')
        if ph_mode and ph_mode in self.data_config:
            state_config['pulse_height'] = self.data_config[ph_mode]

        return state_config

    def apply_state(self, state: Dict[str, Any]):
        """Executes a single mode transition using the robust config.py implementations."""
        state_name = state.get('state_name', 'UNKNOWN')
        logger.info(f"Transitioning to state: {state_name}")

        start_transition = time.time()

        # 1. Stop Data Flow Globally
        stop_params = quabo_driver.DAQ_PARAMS(False, 0, False, False, False)
        self._broadcast_acq_mode(stop_params)
        time.sleep(0.05)  # Fixed hardware settling delay

        # 2. Reconfigure FPGA & MAROCs
        state_data_config = self.prepare_state_config(state)

        if not self.dry_run:
            # We call the functions directly from config.py
            # True indicates we want standard console/logging outputs enabled
            pano_config.do_maroc_config(
                self.modules, self.quabo_uids, self.quabo_info,
                state_data_config, self.obs_config, self.daq_config,
                self.network_config, True
            )

            pano_config.do_mask_config(
                self.modules, state_data_config,
                self.network_config, self.quabo_uids, True
            )

        time.sleep(0.05)  # Fixed hardware settling delay

        # 3. Start Data Flow (New Acquisition Mode)
        start_params = self.build_daq_params(state_data_config)
        self._broadcast_acq_mode(start_params)

        transition_overhead = time.time() - start_transition
        self.stats["total_switch_overhead_sec"] += transition_overhead
        logger.info(f"State {state_name} active. Transition overhead: {transition_overhead:.3f}s")

        # Sleep for observation duration
        duration = state.get('duration_seconds', 1.0)
        time.sleep(duration)

    def run_loop(self):
        """Main interleave execution loop."""
        if not self.states:
            logger.error("No interleave states defined in data_config. Exiting.")
            return

        logger.info("Starting Interleave scheduler...")

        try:
            while self.keep_running:
                for state in self.states:
                    if not self.keep_running:
                        break
                    self.apply_state(state)

                self.stats["total_cycles"] += 1
                if self.max_cycles and self.stats["total_cycles"] >= self.max_cycles:
                    logger.info(f"Reached max cycles ({self.max_cycles}). Stopping.")
                    break
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.shutdown()

    def shutdown(self):
        """Gracefully restores system to the default state and closes connections."""
        logger.info("Restoring pristine default state...")
        # Re-apply the original config
        default_state = {
            'state_name': 'RESTORE_DEFAULT',
            'duration_seconds': 0,
            # We must map back exactly the keys that existed in the original file
            'movie_mode_config': 'image_8bit' if 'image_8bit' in self.data_config else 'image' if 'image' in self.data_config else None,
            'pulse_height_mode_config': 'pulse_height' if 'pulse_height' in self.data_config else None
        }
        self.apply_state(default_state)

        self.executor.shutdown(wait=False)
        for q in self.quabos:
            q.close()
        logger.info("Shutdown complete.")


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