import time
import logging
import sys
import os
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.logging import RichHandler

# Append the control root to python path dynamically based on your tree structure
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ..driver import quabo_driver
from .. import config
from ..utils import config_file, util

from .interleave_helper.pydantic_config_models import load_and_validate_data_config

# --- 1. SETUP LOGGING ---
FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("panoseti.interleave")


# --- 2. EXECUTION ARCHITECTURE ---

class InterleaveController:
    def __init__(self, data_config: dict, obs_config: dict, daq_config: dict,
                 quabo_uids: dict, quabo_info: dict, network_config: dict):
        self.data_config = data_config
        self.obs_config = obs_config
        self.daq_config = daq_config
        self.quabo_uids = quabo_uids
        self.quabo_info = quabo_info
        self.network_config = network_config
        self.interleave_cfg = data_config.get("interleave", {})

        # Extract modules
        self.modules = []
        for dome in obs_config.get('domes', []):
            self.modules.extend(dome.get('modules', []))

        # Instantiate Quabo drivers for direct DAQ control
        self.quabos = []
        for module in self.modules:
            for i in range(4):
                uid = util.quabo_uid(module, quabo_uids, i)
                if not uid:
                    continue
                ip_ports = util.get_quabo_ip_port(module['ip_addr'], i, network_config)
                real_ip = ip_ports['ip_addr']
                cmd_port = ip_ports['cmd_port']
                self.quabos.append(quabo_driver.QUABO(real_ip, cmd_port))

        self.executor = ThreadPoolExecutor(max_workers=len(self.modules) + 4)

    def _broadcast_acq_mode(self, daq_params: quabo_driver.DAQ_PARAMS):
        """Send ACQ mode in parallel to all Quabos"""

        def send_acq(q: quabo_driver.QUABO):
            q.send_daq_params(daq_params)

        futures = [self.executor.submit(send_acq, q) for q in self.quabos]
        for f in as_completed(futures):
            f.result()  # raises exception if network call failed

    def _reconfigure_quabos(self, state_config_dict: dict):
        """
        Reconfigures all quabos based on a synthesized data_config dict for the current state.
        This distributes do_maroc_config across modules concurrently for maximum performance.
        """

        def reconfig_module(module):
            # By passing a [module] list of size 1, we achieve fully parallelized execution while
            # preserving the complex logic inside config.py
            config.do_maroc_config(
                [module], self.quabo_uids, self.quabo_info,
                state_config_dict, self.obs_config, self.daq_config,
                self.network_config, verbose=False
            )

        futures = [self.executor.submit(reconfig_module, module) for module in self.modules]
        for f in as_completed(futures):
            f.result()

    def generate_state_dict(self, state_def: dict) -> dict:
        """
        Creates a mock data_config dictionary that replaces the root 'image'
        and 'pulse_height' keys with the ones requested by the state suffix.
        This tricks the existing config.py software into loading the right mode.
        """
        temp_dict = self.data_config.copy()
        temp_dict.pop('image', None)
        temp_dict.pop('pulse_height', None)

        if state_def.get("movie_mode_config"):
            temp_dict['image'] = self.data_config[state_def["movie_mode_config"]]
        if state_def.get("pulse_height_mode_config"):
            temp_dict['pulse_height'] = self.data_config[state_def["pulse_height_mode_config"]]

        return temp_dict

    def build_daq_params(self, state_dict: dict) -> quabo_driver.DAQ_PARAMS:
        """Construct a quabo_driver.DAQ_PARAMS object based on the current state data config."""
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

    def run_loop(self):
        if not self.interleave_cfg.get("enable", False):
            logger.info("Interleaving disabled. Exiting controller.")
            return

        states = self.interleave_cfg.get("states", [])
        logger.info(f"Starting Interleave Observation Loop with {len(states)} states. Press Ctrl+C to abort.")

        # Data Stop command matching the structure used in config.py's baseline logic
        stop_params = quabo_driver.DAQ_PARAMS(False, 0, False, False, True)

        try:
            while True:
                for state in states:
                    name = state["state_name"]
                    duration = state["duration_seconds"]

                    logger.info(f"--- Entering State: [bold cyan]{name}[/bold cyan] ---", extra={"markup": True})

                    # 1. STOP DAQ FLOW
                    logger.debug("Stopping DAQ packet flow")
                    self._broadcast_acq_mode(stop_params)

                    # 2. RECONFIGURE
                    logger.debug(f"Reconfiguring MAROC and FPGA registers for {name}")
                    state_dict = self.generate_state_dict(state)
                    self._reconfigure_quabos(state_dict)

                    # 3. START DAQ FLOW
                    daq_params = self.build_daq_params(state_dict)
                    logger.debug(f"Starting DAQ packet flow (PH={daq_params.do_ph}, IMG={daq_params.do_image})")
                    self._broadcast_acq_mode(daq_params)

                    # 4. OBSERVE
                    logger.info(f"Observing for {duration} seconds...")
                    time.sleep(duration)

        except KeyboardInterrupt:
            logger.warning("Interleaving aborted by user. Stopping data flow.")
            self._broadcast_acq_mode(stop_params)
        finally:
            self.executor.shutdown(wait=False)
            for q in self.quabos:
                q.close()


if __name__ == "__main__":
    try:
        # Load all observatory configurations utilizing existing mechanism
        obs_config = config_file.get_obs_config()
        daq_config = config_file.get_daq_config()
        quabo_uids = config_file.get_quabo_uids()
        quabo_info = config_file.get_quabo_info()
        network_config = config_file.get_network_config()

        # Load and validate the Data Products configuration fail-fast
        valid_data_config = load_and_validate_data_config("../configs/data_config.json", logger)

        # Initialize and run
        controller = InterleaveController(
            data_config=valid_data_config,
            obs_config=obs_config,
            daq_config=daq_config,
            quabo_uids=quabo_uids,
            quabo_info=quabo_info,
            network_config=network_config
        )
        controller.run_loop()

    except Exception as e:
        logger.error(f"Initialization Failed: {e}", exc_info=True)