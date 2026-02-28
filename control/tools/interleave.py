#!/usr/bin/env python3
"""
interleave.py

PANOSETI Interleaved Observation Controller.
Utilizes a "Cache and Blast" architecture to pre-compute MAROC and FPGA
configurations at startup, eliminating disk/math overhead to achieve
millisecond-precision mode switching during active observations.
"""

import time
import socket
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
    def __init__(self, data_config: Dict[str, Any], obs_config: Dict[str, Any],
                 daq_config: Dict[str, Any], quabo_uids: Dict[str, Any],
                 quabo_info: List[Dict[str, Any]], network_config: Dict[str, Any],
                 dry_run: bool = False, max_cycles: Optional[int] = None):

        self.keep_running = True
        self.dry_run = dry_run
        self.max_cycles = max_cycles
        self.stats = {"total_cycles": 0, "total_switch_overhead_sec": 0.0}

        self._acquire_lock()

        # Freeze original config to guarantee pristine teardown
        self.original_data_config = copy.deepcopy(data_config)
        self.interleave_cfg = self.original_data_config.get("interleave", {})

        self.obs_config = obs_config
        self.daq_config = daq_config
        self.quabo_uids = quabo_uids
        self.quabo_info = quabo_info
        self.network_config = network_config

        self.modules = config_file.get_modules(obs_config)

        self.quabos: Dict[str, quabo_driver.QUABO] = {}
        self.module_ips: List[List[Optional[str]]] = []  # Tracks 0->3 order per module

        for module in self.modules:
            ordered_ips_for_module = []
            for i in range(4):
                uid = util.quabo_uid(module, quabo_uids, i)
                if not uid:
                    ordered_ips_for_module.append(None)
                    continue
                ip_ports = util.get_quabo_ip_port(module['ip_addr'], i, network_config)
                real_ip = ip_ports['ip_addr']
                self.quabos[real_ip] = quabo_driver.QUABO(real_ip, ip_ports['cmd_port'])
                ordered_ips_for_module.append(real_ip)

            self.module_ips.append(ordered_ips_for_module)

        # Thread pool size can now be optimized to the number of modules
        self.executor = ThreadPoolExecutor(max_workers=len(self.modules))

        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)

        self.csv_file, self.csv_writer = None, None
        self._init_event_logger()

        # --- PRE-COMPUTE PHASE ---
        self.state_cache = {}
        self._precompute_state_payloads()

        if self.dry_run:
            logger.info("=== DRY RUN MODE ENABLED ===")

    def _precompute_state_payloads(self) -> None:
        """
        Pre-computes and caches the MAROC dicts, MASK dicts, and DAQ params
        for every interleave state, plus the DEFAULT state.
        """
        logger.info("Pre-computing hardware configurations. This may take a moment...")

        # 1. Compute all schedule states
        for state in self.interleave_cfg.get("states", []):
            name = state["state_name"]
            state_dict = self.generate_state_dict(
                state.get("movie_mode_config"),
                state.get("pulse_height_mode_config")
            )
            self._cache_state(name, state_dict)

        # 2. Compute the pristine original state for teardown
        default_dict = self.generate_state_dict(
            "image" if "image" in self.original_data_config else None,
            "pulse_height" if "pulse_height" in self.original_data_config else None
        )
        self._cache_state("DEFAULT_TEARDOWN", default_dict)
        logger.info("Pre-computation complete. Ready to blast.")

    def _cache_state(self, name: str, state_dict: Dict[str, Any]) -> None:
        maroc_payloads = pano_config.compute_maroc_config(
            self.modules, self.quabo_uids, self.quabo_info,
            state_dict, self.obs_config, self.network_config
        )
        mask_payloads = pano_config.compute_mask_config(
            self.modules, state_dict, self.network_config, self.quabo_uids
        )
        self.state_cache[name] = {
            "maroc": maroc_payloads,
            "mask": mask_payloads,
            "daq": self.build_daq_params(state_dict)
        }

    def _fast_blast_state(self, state_name: str) -> None:
        """
        Parallel across modules, but STRICTLY SEQUENTIAL (0, 1, 2, 3) within each module.
        Paced to prevent UDP buffer overflows on the Quabo embedded processors.
        """
        cache = self.state_cache[state_name]

        if self.dry_run:
            logger.info(f"[DRY-RUN] Simulating fast-blast for state: {state_name}")
            return

        def blast_module(ordered_ips: List[Optional[str]]):
            for ip in ordered_ips:
                if not ip: continue
                q = self.quabos[ip]

                original_timeout = q.sock.gettimeout()
                q.sock.settimeout(0.01)

                try:
                    # 1. Send MAROC Configurations
                    if ip in cache['maroc']:
                        for m_dict in cache['maroc'][ip]:
                            try:
                                q.send_maroc_params(m_dict)
                                time.sleep(0.01)  # <-- CRITICAL: Give FPGA time to clock MAROC chips
                            except socket.timeout:
                                pass
                            except Exception as e:
                                logger.debug(f"Ignored non-timeout error on MAROC send: {e}")

                    # 2. Send FPGA Trigger Masks
                    # if ip in cache['mask']:
                    #     try:
                    #         q.send_trigger_mask(cache['mask'][ip])
                    #         time.sleep(0.01)  # <-- CRITICAL: Pacing
                    #         q.send_goe_mask(cache['mask'][ip])
                    #         time.sleep(0.01)  # <-- CRITICAL: Pacing
                    #     except socket.timeout:
                    #         pass

                    # 3. Send DAQ Configuration (start data flow)
                    try:
                        q.send_daq_params(cache['daq'])
                        time.sleep(0.01)  # <-- CRITICAL: Pacing
                    except socket.timeout:
                        pass

                finally:
                    # Always safely restore the original timeout
                    q.sock.settimeout(original_timeout)

        # Execute concurrently across modules
        futures = [self.executor.submit(blast_module, ips) for ips in self.module_ips]
        for f in as_completed(futures):
            f.result()

    def _stop_data_flow(self) -> None:
        """Stops DAQ sequentially within modules for hardware safety."""
        stop_params = quabo_driver.DAQ_PARAMS(False, 0, False, False, True)
        if self.dry_run: return

        def stop_module(ordered_ips: List[Optional[str]]):
            for ip in ordered_ips:
                if ip:
                    self.quabos[ip].send_daq_params(stop_params)

        futures = [self.executor.submit(stop_module, ips) for ips in self.module_ips]
        for f in as_completed(futures): f.result()

    # --- Utility Methods ---
    def _acquire_lock(self):
        if os.path.exists(PID_FILE):
            with open(PID_FILE, "r") as f:
                try:
                    old_pid = int(f.read().strip())
                    os.kill(old_pid, 0)
                    logger.critical(f"Another process running (PID {old_pid}). Run `config.py --stop-interleave`.")
                    sys.exit(1)
                except:
                    pass
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))

    def _release_lock(self):
        if os.path.exists(PID_FILE): os.remove(PID_FILE)

    def _handle_shutdown_signal(self, signum, frame):
        logger.warning("Shutdown signal received.")
        self.keep_running = False

    def _init_event_logger(self):
        os.makedirs(os.path.dirname(EVENT_LOG_FILE), exist_ok=True)
        exists = os.path.exists(EVENT_LOG_FILE) and os.path.getsize(EVENT_LOG_FILE) > 0
        self.csv_file = open(EVENT_LOG_FILE, mode='a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if not exists:
            self.csv_writer.writerow(["unix_timestamp", "utc_datetime", "event_type", "state_name", "details"])

    def _log_event(self, event_type: str, state_name: str, details: str = ""):
        now_ts = time.time()
        now_utc = datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat()
        if self.csv_writer and self.csv_file:
            self.csv_writer.writerow([f"{now_ts:.6f}", now_utc, event_type, state_name, details])
            self.csv_file.flush()
            os.fsync(self.csv_file.fileno())

    def generate_state_dict(self, movie_key: Optional[str], ph_key: Optional[str]) -> Dict[str, Any]:
        temp_dict = copy.deepcopy(self.original_data_config)
        temp_dict.pop('image', None);
        temp_dict.pop('pulse_height', None)
        if movie_key and movie_key in temp_dict: temp_dict['image'] = copy.deepcopy(temp_dict[movie_key])
        if ph_key and ph_key in temp_dict: temp_dict['pulse_height'] = copy.deepcopy(temp_dict[ph_key])
        return temp_dict

    def build_daq_params(self, state_dict: Dict[str, Any]) -> quabo_driver.DAQ_PARAMS:
        do_img, do_ph = 'image' in state_dict, 'pulse_height' in state_dict
        image_us = state_dict['image'].get('integration_time_usec', 0) if do_img else 0
        image_8bit = (state_dict['image'].get('quabo_sample_size', 0) == 8) if do_img else False
        any_trig, grp_ph = False, False
        if do_ph and 'any_trigger' in state_dict['pulse_height']:
            any_trig = True
            grp_ph = bool(state_dict['pulse_height']['any_trigger'].get('group_ph_frames', 0))
        return quabo_driver.DAQ_PARAMS(do_img, image_us, image_8bit, do_ph, True, any_trig, grp_ph)

    def _sleep_until(self, target_time: float):
        while self.keep_running:
            rem = target_time - time.perf_counter()
            if rem <= 0.005: break
            time.sleep(rem - 0.005)
        while self.keep_running and time.perf_counter() < target_time: pass

    # --- Core Loop ---
    def run_loop(self) -> None:
        if not self.interleave_cfg.get("enable", False):
            logger.info("Interleaving disabled. Exiting.")
            self._release_lock()
            return

        states = self.interleave_cfg.get("states", [])
        schedule_start_time = time.perf_counter()
        next_state_time = schedule_start_time

        self._log_event("INTERLEAVE_START", "GLOBAL", "Pre-computation complete. Running schedule.")

        try:
            while self.keep_running:
                if self.max_cycles and self.stats["total_cycles"] >= self.max_cycles: break

                for state in states:
                    if not self.keep_running: break
                    name = state["state_name"]
                    duration = state["duration_seconds"]
                    next_state_time += duration

                    logger.info(f"\n--- Entering State: {name} ---")
                    self._log_event("SWITCH_START", name, "Stopping DAQ and fast-blasting cache")
                    t_overhead_start = time.perf_counter()

                    self._stop_data_flow()
                    self._fast_blast_state(name)  # Sub-100ms switch

                    overhead = time.perf_counter() - t_overhead_start
                    self.stats["total_switch_overhead_sec"] += overhead
                    self._log_event("OBSERVE_START", name, f"Reconfigured in {overhead:.3f}s")

                    if time.perf_counter() > next_state_time:
                        logger.warning("Overhead exceeded state duration. Resetting timeline.")
                        next_state_time = time.perf_counter()
                        continue

                    self._sleep_until(next_state_time)
                    self._log_event("OBSERVE_END", name, "Observation duration complete")

                self.stats["total_cycles"] += 1

        except Exception as e:
            logger.error(f"Error in interleaving loop: {e}", exc_info=True)
            self._log_event("ERROR", "GLOBAL", str(e))
        finally:
            self._teardown()

    def _teardown(self) -> None:
        logger.info("Teardown initiated. Restoring pristine hardware defaults...")
        self._log_event("TEARDOWN_START", "GLOBAL", "Restoring pristine default state from RAM cache")
        try:
            self._stop_data_flow()
            self._fast_blast_state("DEFAULT_TEARDOWN")
            logger.info("Hardware defaults restored successfully.")
            self._log_event("TEARDOWN_COMPLETE", "GLOBAL", "Done")
        except Exception as e:
            logger.error(f"Failed to restore hardware: {e}")
            self._log_event("TEARDOWN_ERROR", "GLOBAL", str(e))
        finally:
            self.executor.shutdown(wait=False)
            if not self.dry_run:
                for q in self.quabos.values(): q.close()
            if self.csv_file: self.csv_file.close()
            self._release_lock()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-cycles', type=int, default=None)
    args = parser.parse_args()

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
        logger.error(f"Startup failed: {e}")
        sys.exit(1)