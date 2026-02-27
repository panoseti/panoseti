
import time
import logging
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.logging import RichHandler

# Mock imports based on the PANOSETI architecture described
from ..driver import quabo_driver
# import config as pano_config

from interleave_helper.pydantic_config_models import load_and_validate_data_config

# --- 1. SETUP LOGGING ---
FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("panoseti.interleave")


# --- 3. EXECUTION ARCHITECTURE ---

class InterleaveController:
    def __init__(self, quabo_ips: List[str], data_config: dict):
        self.data_config = data_config
        self.interleave_cfg = data_config.get("interleave", {})

        # Instantiate Quabo drivers (Mocked here, replace with actual quabo_driver.QUABO)
        self.quabos = [quabo_driver.QUABO(ip) for ip in quabo_ips]
        self.quabos = quabo_ips  # Mocking IPs as targets for now

        self.executor = ThreadPoolExecutor(max_workers=len(quabo_ips) + 4)

    def _broadcast_acq_mode(self, mode: int):
        """Send ACQ mode in parallel to all Quabos"""

        def send_acq(q_ip):
            # MOCK: q.send_acq_paramaters()
            # print(f"Sending ACQ={mode} to {q_ip}")
            pass

        futures = [self.executor.submit(send_acq, q) for q in self.quabos]
        for f in as_completed(futures):
            f.result()  # raises exception if network call failed

    def _reconfigure_quabos(self, state_config_dict: dict):
        """
        Reconfigures all quabos based on a synthesized data_config dict for the current state.
        This allows reuse of existing do_maroc_config logic.
        """

        def reconfig(q_ip):
            # MOCK: pano_config.do_maroc_config(..., state_config_dict, ...)
            pass

        futures = [self.executor.submit(reconfig, q) for q in self.quabos]
        for f in as_completed(futures):
            f.result()

    def generate_state_dict(self, state_def: dict) -> dict:
        """
        Creates a mock data_config dictionary that replaces the root 'image'
        and 'pulse_height' keys with the ones requested by the state suffix.
        This tricks the existing software into loading the right mode.
        """
        temp_dict = self.data_config.copy()

        # Wipe base modes
        temp_dict.pop('image', None)
        temp_dict.pop('pulse_height', None)

        # Inject target modes as the new base modes
        if state_def.get("movie_mode_config"):
            temp_dict['image'] = self.data_config[state_def["movie_mode_config"]]

        if state_def.get("pulse_height_mode_config"):
            temp_dict['pulse_height'] = self.data_config[state_def["pulse_height_mode_config"]]

        return temp_dict

    def run_loop(self):
        if not self.interleave_cfg.get("enable", False):
            logger.info("Interleaving disabled. Exiting controller.")
            return

        states = self.interleave_cfg.get("states", [])
        logger.info(f"Starting Interleave Observation Loop with {len(states)} states. Press Ctrl+C to abort.")

        try:
            while True:
                for state in states:
                    name = state["state_name"]
                    duration = state["duration_seconds"]

                    logger.info(f"--- Entering State: [bold cyan]{name}[/bold cyan] ---", extra={"markup": True})

                    # 1. STOP DAQ FLOW
                    logger.debug("Stopping DAQ packet flow (ACQ=0)")
                    self._broadcast_acq_mode(0)

                    # 2. RECONFIGURE
                    logger.debug(f"Reconfiguring MAROC and FPGA registers for {name}")
                    state_dict = self.generate_state_dict(state)
                    self._reconfigure_quabos(state_dict)

                    # 3. START DAQ FLOW
                    # ACQ mode determination based on active configs (simplified logic)
                    # Real logic will rely on your ACQ mask building function
                    target_acq_mode = 0x01 if state.get("pulse_height_mode_config") else 0x02
                    logger.debug(f"Starting DAQ packet flow (ACQ={target_acq_mode})")
                    self._broadcast_acq_mode(target_acq_mode)

                    # 4. OBSERVE
                    logger.info(f"Observing for {duration} seconds...")
                    time.sleep(duration)

        except KeyboardInterrupt:
            logger.warning("Interleaving aborted by user. Stopping data flow.")
            self._broadcast_acq_mode(0)
        finally:
            self.executor.shutdown(wait=False)


if __name__ == "__main__":
    # Example usage
    try:
        # Load and validate
        valid_config = load_and_validate_data_config("data_config.json", logger)

        # Mock Quabo IPs found via standard observatory_config parsing
        active_quabo_ips = ["192.168.3.248", "192.168.3.249", "192.168.3.250"]

        # Run
        controller = InterleaveController(active_quabo_ips, valid_config)
        controller.run_loop()

    except Exception as e:
        logger.error(f"Initialization Failed: {e}")