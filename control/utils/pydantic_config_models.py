"""
pydantic_config_models.py

Centralized Pydantic models for validating PANOSETI configuration files.
"""

import logging
from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, model_validator, ConfigDict, IPvAnyAddress, field_validator
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
    """Allows extra fields specifically for dynamic image_* and pulse_height_* keys."""
    run_type: str = Field(..., max_length=MAX_RUN_TYPE_LENGTH)
    detector_overvoltage: Literal[2, 3] = Field(
        description="over voltage used for the observation. "
                    "For now, we only have calibration data for 2V and 3V."
    )
    max_file_size_mb: int = Field(..., gt=0)
    gain: Optional[float] = None

    # Reserved default states & hardware params
    image: Optional[ImageMode] = None
    pulse_height: Optional[PulseHeightMode] = None
    long_pulse: Optional[LongPulseMode] = None
    flash_params: Optional[FlashParams] = None
    stim_params: Optional[StimParams] = None

    model_config = ConfigDict(extra='allow')

    @field_validator("run_type")
    def validate_run_type(cls, v):
        if any(ch in INVALID_RUN_TYPE_CHARS for ch in v):
            raise ValueError(f"Invalid run_type: '{v}' contains at least one invalid character: {INVALID_RUN_TYPE_CHARS}")
        return v

    @model_validator(mode='after')
    def validate_interleave_and_exclusions(self) -> 'DataConfigValidator':
        extra_data = self.model_extra or {}
        ph_configs: Dict[str, PulseHeightMode] = {}
        img_configs: Dict[str, ImageMode] = {}

        if self.pulse_height: ph_configs["pulse_height"] = self.pulse_height
        if self.image: img_configs["image"] = self.image

        for key, val in extra_data.items():
            if key == "interleave": continue
            if key.startswith("pulse_height_"): ph_configs[key] = PulseHeightMode(**val)
            elif key.startswith("image_"): img_configs[key] = ImageMode(**val)
            else: raise ValueError(f"Invalid root key '{key}'. Must be 'image_*' or 'pulse_height_*'.")

        interleave_data = extra_data.get("interleave")
        if not interleave_data or not interleave_data.get("enable", False):
            if self.image and self.pulse_height:
                if self.pulse_height.two_pixel_trigger > 0 or self.pulse_height.three_pixel_trigger > 0:
                    raise ValueError("MUTUAL EXCLUSION VIOLATION: Cannot enable default image mode while multi-pixel triggers are enabled.")
            return self

        interleave = InterleaveConfig(**interleave_data)
        for state in interleave.states:
            if state.movie_mode_config and state.movie_mode_config not in img_configs:
                raise ValueError(f"Missing movie_mode_config '{state.movie_mode_config}'")
            if state.pulse_height_mode_config and state.pulse_height_mode_config not in ph_configs:
                raise ValueError(f"Missing pulse_height_mode_config '{state.pulse_height_mode_config}'")
            if state.movie_mode_config and state.pulse_height_mode_config:
                ph_mode = ph_configs[state.pulse_height_mode_config]
                if ph_mode.two_pixel_trigger > 0 or ph_mode.three_pixel_trigger > 0:
                    raise ValueError(f"MUTUAL EXCLUSION: Image mode vs Multi-pixel trigger in state '{state.state_name}'.")
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

class DaqNodeValidator(BaseStrictModel):
    comment: Optional[str]
    username: str
    data_dir: str
    ip_addr: IPvAnyAddress
    # Allow List[int] because `expand_ranges()` parses "253" into a list prior to validation
    module_ids: Union[str, List[int]]
    bindhost: Optional[str] = Field("0.0.0.0")

class DaqConfigValidator(BaseStrictModel):
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