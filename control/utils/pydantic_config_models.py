"""
pydantic_config_models.py

Centralized Pydantic models for validating PANOSETI configuration files.
"""

import logging
from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import (
    BaseModel, Field, model_validator,
    ConfigDict, IPvAnyAddress, field_validator,
    ValidationError
)
from rich.console import Console
from rich.pretty import pprint

console = Console()

# Global restrictions
## Data config
MAX_RUN_TYPE_LENGTH = 14
INVALID_RUN_TYPE_CHARS = [".", "_", " ", ]

MIN_PULSE_HEIGHT_PE_THRESHOLD = 2.0
MIN_MOVIE_MODE_PE_THRESHOLD = 1.0


# --- Shared & Base Models ---

class BaseStrictModel(BaseModel):
    """Disallows extra fields to catch typos in configuration keys."""
    model_config = ConfigDict(extra='forbid')

# --- Data Config Models ---

class AnyTriggerConfig(BaseStrictModel):
    group_ph_frames: int = Field(0, description="If set to 1, hashpipe will group 4 packets from 4 quabos.")

class PulseHeightMode(BaseStrictModel):
    pe_threshold: float = Field(..., ge=MIN_PULSE_HEIGHT_PE_THRESHOLD, description="Pulse height threshold in photoelectrons")
    any_trigger: Optional[AnyTriggerConfig] = None
    two_pixel_trigger: int = Field(0, description="If set to 1, 2 pixel trigger mode will be enabled.")
    three_pixel_trigger: int = Field(0, description="If set to 1, 3 pixel trigger mode will be enabled.")


class ImageMode(BaseStrictModel):
    integration_time_usec: int = Field(..., ge=20, description="Integration time in microseconds")
    pe_threshold: float = Field(..., ge=MIN_MOVIE_MODE_PE_THRESHOLD, description="Image mode threshold in photoelectrons")
    quabo_sample_size: Literal[8, 16] = Field(..., description="Size of the sample")
    quabo_num: Optional[int] = Field(None, description="Omit for all 4 quabos")

class LongPulseMode(BaseStrictModel):
    octaves: int
    threshold_sigma: List[float]

class FlashParams(BaseStrictModel):
    rate: int = Field(..., ge=0, le=7, description="3-bit value controlling flash rate (0-7)")
    level: int = Field(..., ge=0, le=31, description="Controls DC supply level (0-31)")
    width: int = Field(..., ge=0, le=15, description="Controls pulse width (0-15)")

class StimParams(BaseStrictModel):
    rate: int = Field(..., ge=0, le=7)
    level: int = Field(..., ge=0, le=255)
    mask: List[int] = Field(..., max_length=4, min_length=4)

class InterleaveState(BaseStrictModel):
    state_name: str
    duration_seconds: float = Field(..., gt=0.0)
    movie_mode_config: Optional[str] = None
    pulse_height_mode_config: Optional[str] = None

    @model_validator(mode='after')
    def check_at_least_one_mode(self) -> 'InterleaveState':
        if not self.movie_mode_config and not self.pulse_height_mode_config:
            raise ValueError(f"State '{self.state_name}' must have at least one valid mode (movie or pulse_height).")
        return self

class InterleaveConfig(BaseStrictModel):
    enable: bool = Field(False)
    states: List[InterleaveState] = Field([])


class DataConfigValidator(BaseModel):
    # We must use extra='allow' so Pydantic parses them, but we will
    # strictly validate the extra keys dynamically in mode='after'.
    model_config = ConfigDict(extra='allow')

    run_type: str = Field(..., max_length=MAX_RUN_TYPE_LENGTH)
    detector_overvoltage: Optional[int] = None
    gain: Optional[int] = None
    max_file_size_mb: Optional[int] = None
    image: Optional[ImageMode] = None
    pulse_height: Optional[PulseHeightMode] = None
    interleave: Optional[Any] = None  # Assuming you have an InterleaveConfig model
    stim_params: Optional[Any] = None
    flash_params: Optional[Any] = None

    @model_validator(mode='after')
    def validate_dynamic_modes_and_interleave(self):
        dynamic_keys = []
        ph_modes_dict = {}

        if self.pulse_height:
            ph_modes_dict['pulse_height'] = self.pulse_height

        # Global hardware check for the default modes
        if self.image and self.pulse_height:
            if self.pulse_height.two_pixel_trigger > 0 or self.pulse_height.three_pixel_trigger > 0:
                raise ValueError(
                    "Hardware Constraint Violation: Cannot enable root 'image' mode while "
                    "root 'pulse_height' mode has two_pixel_trigger or three_pixel_trigger > 0."
                )

        if self.model_extra:
            for key, val in self.model_extra.items():
                if key.startswith('image_'):
                    try:
                        ImageMode(**val)
                        dynamic_keys.append(key)
                    except ValidationError as e:
                        raise ValueError(f"Invalid fields in dynamic mode '{key}': {e}")
                elif key.startswith('pulse_height_'):
                    try:
                        ph_obj = PulseHeightMode(**val)
                        ph_modes_dict[key] = ph_obj
                        dynamic_keys.append(key)
                    except ValidationError as e:
                        raise ValueError(f"Invalid fields in dynamic mode '{key}': {e}")
                else:
                    raise ValueError(f"Unrecognized configuration key or typo detected: '{key}'")

        if self.interleave and getattr(self.interleave, 'states', None):
            valid_image_modes = ['image'] + dynamic_keys
            valid_ph_modes = ['pulse_height'] + dynamic_keys

            for state in self.interleave.states:
                m_conf = state.movie_mode_config
                p_conf = state.pulse_height_mode_config

                # Rule 1: No Double Nulls
                if not m_conf and not p_conf:
                    raise ValueError(
                        f"Interleave state '{state.state_name}' has BOTH movie_mode and pulse_height_mode set to null. At least one must be active.")

                if m_conf and m_conf not in valid_image_modes:
                    raise ValueError(f"Interleave state '{state.state_name}' references missing movie mode: '{m_conf}'")

                if p_conf and p_conf not in valid_ph_modes:
                    raise ValueError(
                        f"Interleave state '{state.state_name}' references missing pulse height mode: '{p_conf}'")

                # Rule 2: Hardware Mutual Exclusion
                if m_conf and p_conf:
                    ph_obj = ph_modes_dict.get(p_conf)
                    if ph_obj and (ph_obj.two_pixel_trigger > 0 or ph_obj.three_pixel_trigger > 0):
                        raise ValueError(
                            f"Hardware Constraint Violation in interleave state '{state.state_name}': "
                            f"Cannot enable movie-mode ('{m_conf}') while pulse height mode ('{p_conf}') "
                            f"has two_pixel_trigger or three_pixel_trigger enabled."
                        )

        return self


# --- Obs Config Models ---

class WpsConfig(BaseStrictModel):
    url: str
    quabo_socket: int


class ObsModuleConfig(BaseStrictModel):
    mobo_serialno: str
    quabo_version: Union[str, List[str]]
    ip_addr: IPvAnyAddress
    wps: Optional[str] = None
    ups: Optional[str] = None
    timing_mode: Optional[str] = Field("wr", pattern="^(wr|gnss)$")
    azimuth: Optional[float] = Field(None, ge=0, le=360)
    elevation: Optional[float] = Field(None, ge=-90, le=90)
    position_angle: Optional[float] = None

    # Injected fields by config_file.py at runtime
    id: Optional[int] = None


class ObsDomeConfig(BaseStrictModel):
    name: str
    obslat: float = Field(..., ge=-90, le=90)
    obslon: float = Field(..., ge=-180, le=180)
    obsalt: float
    modules: List[ObsModuleConfig]

    # Injected fields by config_file.py at runtime
    num: Optional[int] = None


class ObsConfigValidator(BaseModel):
    name: str
    comment: Optional[str] = None
    wr_ip_addr: Optional[IPvAnyAddress] = Field("192.168.1.254")
    dome_controller_ip_addr: Optional[IPvAnyAddress] = None
    gps_port: Optional[str] = Field("/dev/ttyUSB0")
    detector_overvoltage: Optional[int] = None
    domes: List[ObsDomeConfig]

    model_config = ConfigDict(extra='allow')

    @model_validator(mode='after')
    def validate_wps_extras(self) -> 'ObsConfigValidator':
        """Ensures that dynamic 'wps-' keys match the WpsConfig schema."""
        extra_data = self.model_extra or {}
        for key, val in extra_data.items():
            if key.startswith("wps"):
                try:
                    WpsConfig(**val)
                except Exception as e:
                    raise ValueError(f"Invalid format for '{key}': {e}")
            else:
                raise ValueError(f"Extra key '{key}' is not allowed unless it's a 'wps' unit.")
        return self

# --- DAQ Config Models ---

class CompactList(list):
    """A list that prints compactly in the terminal, preventing long vertical scrolls."""
    def __repr__(self):
        if len(self) > 6:
            return f"[{self[0]}, {self[1]}, ..., {self[-2]}, {self[-1]}] (len={len(self)})"
        return super().__repr__()

class DaqNodeValidator(BaseStrictModel):
    username: str
    data_dir: str
    ip_addr: IPvAnyAddress
    module_ids: Union[str, List[int]]
    bindhost: Optional[str] = Field("0.0.0.0")

    @field_validator('module_ids', mode='after')
    @classmethod
    def compact_module_ids(cls, v):
        """Converts the parsed list into a CompactList so Pydantic dumps it cleanly on one line."""
        if isinstance(v, list):
            return CompactList(v)
        return v

class DaqConfigValidator(BaseStrictModel):
    comment: Optional[str] = None
    head_node_data_dir: str
    head_node_ip_addr: IPvAnyAddress
    head_node_container: bool = Field(False)
    daq_nodes: List[DaqNodeValidator]

# --- Network Config Models ---

class PortForwarding(BaseStrictModel):
    status: bool
    gw_ip: IPvAnyAddress
    reboot_port: Optional[List[int]] = None
    cmd_port: Optional[List[int]] = None
    port: Optional[int] = None

class NetworkModule(BaseStrictModel):
    ip_addr: IPvAnyAddress
    port_forwarding: PortForwarding

class NetworkDaqNode(BaseStrictModel):
    ip_addr: IPvAnyAddress
    port_forwarding: PortForwarding

class NetworkConfigValidator(BaseStrictModel):
    modules: List[NetworkModule]
    daq_nodes: List[NetworkDaqNode]

# --- Daemon Config Models ---

class Daemons(BaseModel):
    model_config = ConfigDict(extra='allow') # Allow dynamic casper_xx keys

class DaemonConfigValidator(BaseStrictModel):
    daemons: Daemons

# --- Firmware Config Models ---

class FirmwareConfigValidator(BaseModel):
    model_config = ConfigDict(extra='allow') # Allow 'qfp', 'bga', or future hardware variants

# --- Quabo UIDs Config Models ---
class QuaboUidEntry(BaseStrictModel):
    uid: str = Field(..., description="Hex string of the Quabo UID. Empty string if offline.")

class QuaboUidModule(BaseStrictModel):
    quabos: List[QuaboUidEntry] = Field(..., min_length=4, max_length=4)

class QuaboUidDome(BaseStrictModel):
    modules: List[QuaboUidModule]

class QuaboUidsValidator(BaseStrictModel):
    domes: List[QuaboUidDome]