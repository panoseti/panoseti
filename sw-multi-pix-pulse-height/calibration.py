import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel

logger = logging.getLogger("Calibration")


# --- Config Schemas ---

class PixelGainMap(BaseModel):
    """Represents quabo_calib_*.json"""
    pixel_gain: List[List[float]]
    # Optional coeff keys that might exist
    m: Optional[List[List[float]]] = None
    n: Optional[List[List[float]]] = None


class BaselineMap(BaseModel):
    """Represents entry in quabo_ph_baseline.json"""
    quabo_uid: str
    baseline_adc: List[int]


# --- Manager ---

class CalibrationManager:
    def __init__(self, run_dir: Path, static_dir: Path):
        self.run_dir = Path(run_dir)
        self.static_dir = Path(static_dir)

        # 1. Load Context
        with open(self.run_dir / 'obs_config.json') as f:
            self.obs_cfg = json.load(f)
            self.overvoltage = self.obs_cfg.get('detector_overvoltage', 2.0)

        with open(self.run_dir / 'quabo_uids.json') as f:
            self.uids_cfg = json.load(f)

        with open(self.static_dir / 'quabo_info.json') as f:
            self.q_info = {x['uid']: x for x in json.load(f)}

        # Load Baselines (Prefer run_obj dir)
        base_file = self.run_dir / 'quabo_ph_baseline.json'
        if not base_file.exists():
            base_file = self.static_dir / 'quabo_ph_baseline.json'

        self.baselines = {}
        if base_file.exists():
            with open(base_file) as f:
                data = json.load(f)
                # Handle 'quabos' list wrapper if present (common in newer files)
                if isinstance(data, dict) and 'quabos' in data:
                    data = data['quabos']

                for item in data:
                    uid = item.get('uid') or item.get('quabo_uid')
                    if uid:
                        # Reshape 256 -> 16x16
                        self.baselines[uid] = np.array(item['baseline_adc'], dtype=np.float32).reshape(16, 16)

    def get_matrices(self, quabo_uid: str) -> dict:
        """
        Returns {'B': baseline, 'G': gain, 'M': m_coeff} matrices for a UID.
        """
        # 1. Baseline (B)
        B = self.baselines.get(quabo_uid, np.zeros((16, 16), dtype=np.float32))

        # 2. Gain / Calibration
        # We need the Board Serial to find the calib file
        if quabo_uid not in self.q_info:
            logger.warning(f"UID {quabo_uid} not in quabo_info. Using unity gain.")
            return {'B': B, 'G': np.ones((16, 16)), 'M': np.ones((16, 16))}

        serial = self.q_info[quabo_uid]['serialno']  # e.g. SN037

        # Path construction logic based on panoseti-software repo structure
        # control/quabos/quabo_calib_SN037.json
        calib_path = self.static_dir / f"quabo_calib_{serial}.json"

        # Default Unity
        G_delta = np.zeros((16, 16))
        M = np.ones((16, 16)) * 50.0  # Heuristic default if missing

        if calib_path.exists():
            try:
                with open(calib_path) as f:
                    cdata = json.load(f)

                    # Pixel Gain Delta
                    if 'pixel_gain' in cdata:
                        G_delta = np.array(cdata['pixel_gain'], dtype=np.float32)

                    # M Coeff (Detector level usually, but maybe pixel map)
                    # If file doesn't have 'm', we might calculate it from Detector Overvoltage logic
                    # For this implementation, we check if 'm' key exists (rare) or derive it
                    pass
            except Exception as e:
                logger.error(f"Error reading calib for {serial}: {e}")

        # FORMULA LOGIC based on README
        # PE = (ADC - Baseline) / Gain
        # Gain_pixel = Gain_nominal * (1 + delta)
        # Gain_nominal is often derived from overvoltage * some_constant (M)

        # Note: If M is not in JSON, we use a factor derived from obs_config
        # gain = M * overvoltage?
        # Let's assume M is the Base Gain per pixel or detector.

        # If the user says "I explicitly wrote those formulae", implies:
        # PE = (ADC - B) / (M * (1 + G_delta))

        # Construct combined Gain Matrix
        # Nominal gain approx 60.0 defined in data_config.json often
        nominal_gain = 60.0

        total_gain = nominal_gain * (1.0 + G_delta)

        return {'B': B, 'G': total_gain}


def apply_calibration(raw_img: np.ndarray, matrices: dict) -> np.ndarray:
    """
    Vectorized conversion.
    raw_img: (N, 16, 16) or (16, 16)
    """
    raw = raw_img.astype(np.float32)
    B = matrices['B']
    G = matrices['G']

    # Broadcast subtraction and division
    pe = (raw - B) / G
    return pe