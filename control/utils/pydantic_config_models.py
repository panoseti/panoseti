"""
pydantic_config_models.py

Centralized Pydantic models for validating PANOSETI configuration files.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    IPvAnyAddress,
    ValidationError,
    field_validator,
    model_validator,
)
from rich.console import Console

console = Console()

# Global restrictions
## Data config
MAX_RUN_TYPE_LENGTH = 14
INVALID_RUN_TYPE_CHARS = [".", "_", " ", ]

MIN_PULSE_HEIGHT_PE_THRESHOLD = 2.0
MIN_MOVIE_MODE_PE_THRESHOLD = 1.0


# ---------------------------
# ---Shared & Base Models ---
# ---------------------------

class BaseStrictModel(BaseModel):
    """Disallows extra fields to catch typos in configuration keys."""
    model_config = ConfigDict(extra='forbid')

# --------------------------
# --- Data Config Models ---
# --------------------------

## -- data_config: Pulse-height mode
class AnyTriggerConfig(BaseStrictModel):
    group_ph_frames: int = Field(0, description="If set to 1, hashpipe will group 4 packets from 4 quabos.")

class PulseHeightMode(BaseStrictModel):
    pe_threshold: float = Field(..., ge=MIN_PULSE_HEIGHT_PE_THRESHOLD, description="Pulse height threshold in photoelectrons")
    any_trigger: AnyTriggerConfig | None = None
    two_pixel_trigger: int = Field(0, description="If set to 1, 2 pixel trigger mode will be enabled.")
    three_pixel_trigger: int = Field(0, description="If set to 1, 3 pixel trigger mode will be enabled.")


## -- data_config: Movie-mode
class ImageMode(BaseStrictModel):
    integration_time_usec: int = Field(..., ge=20, description="Integration time in microseconds")
    pe_threshold: float = Field(..., ge=MIN_MOVIE_MODE_PE_THRESHOLD, description="Image mode threshold in photoelectrons")
    quabo_sample_size: Literal[8, 16] = Field(..., description="Size of the sample")

    @field_validator('integration_time_usec')
    def check_integration_time_divisor(cls, v):
        if 1_000_000 % v != 0:
            raise ValueError(f"integration_time_usec ({v}) must evenly divide 1,000,000 usec.")
        return v

## -- data_config: Test signal injection
class LongPulseMode(BaseStrictModel):
    octaves: int
    threshold_sigma: list[float]

class FlashParams(BaseStrictModel):
    rate: int = Field(..., ge=0, le=7, description="3-bit value controlling flash rate (0-7)")
    level: int = Field(..., ge=0, le=31, description="Controls DC supply level (0-31)")
    width: int = Field(..., ge=0, le=15, description="Controls pulse width (0-15)")

class StimParams(BaseStrictModel):
    rate: int = Field(..., ge=0, le=7, description="Rate from 190 to 24,400 Hz")
    level: int = Field(..., ge=0, le=255)
    mask: list[bool] = Field(..., max_length=4, min_length=4)

## data_config: Interleaving mode
class InterleaveState(BaseStrictModel):
    state_name: str
    duration_seconds: float = Field(..., gt=0.01)
    movie_mode_config: str | None = None
    pulse_height_mode_config: str | None = None

    @model_validator(mode='after')
    def check_at_least_one_mode(self) -> InterleaveState:
        if not self.movie_mode_config and not self.pulse_height_mode_config:
            raise ValueError(f"State '{self.state_name}' must have at least one valid mode (movie or pulse_height).")
        return self

class InterleaveConfig(BaseStrictModel):
    enable: bool = Field(False)
    states: list[InterleaveState] = Field([])

## data_config: global validator
class DataConfigValidator(BaseModel):
    # We must use extra='allow' so Pydantic parses them, but we will
    # strictly validate the extra keys dynamically in mode='after'.
    model_config = ConfigDict(extra='allow')

    run_type: str = Field(..., max_length=MAX_RUN_TYPE_LENGTH)
    detector_overvoltage: Literal[2, 3] | None = None
    gain: int | None = None
    max_file_size_mb: int | None = Field(None, gt=0)
    image: ImageMode | None = None
    pulse_height: PulseHeightMode | None = None
    interleave: InterleaveConfig | None = None  # Assuming you have an InterleaveConfig model
    stim_params: StimParams | None = None
    flash_params: FlashParams | None = None

    @field_validator("run_type")
    def validate_run_type(cls, v):
        if any(ch in INVALID_RUN_TYPE_CHARS for ch in v):
            raise ValueError(
                f"Invalid run_type: '{v}' contains at least one invalid character: {INVALID_RUN_TYPE_CHARS}")
        return v

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
                        raise ValueError(f"Invalid fields in dynamic mode '{key}': {e}") from e
                elif key.startswith('pulse_height_'):
                    try:
                        ph_obj = PulseHeightMode(**val)
                        ph_modes_dict[key] = ph_obj
                        dynamic_keys.append(key)
                    except ValidationError as e:
                        raise ValueError(f"Invalid fields in dynamic mode '{key}': {e}") from e
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

# -------------------------
# --- Obs Config Models ---
# -------------------------

class WpsConfig(BaseStrictModel):
    url: str
    quabo_socket: int


class ObsModuleConfig(BaseStrictModel):
    mobo_serialno: str
    quabo_version: str | list[str]
    ip_addr: IPvAnyAddress
    wps: str | None = None
    ups: str | None = None
    timing_mode: str | None = Field("wr", pattern="^(wr|gnss)$")
    azimuth: float | None = Field(None, ge=0, le=360)
    elevation: float | None = Field(None, ge=-90, le=90)
    position_angle: float | None = None

    # Injected fields by config_file.py at runtime
    id: int | None = None


class ObsDomeConfig(BaseStrictModel):
    name: str
    obslat: float = Field(..., ge=-90, le=90)
    obslon: float = Field(..., ge=-180, le=180)
    obsalt: float
    modules: list[ObsModuleConfig]

    # Injected fields by config_file.py at runtime
    num: int | None = None


class ObsConfigValidator(BaseModel):
    name: str
    comment: str | None = None
    wr_ip_addr: IPvAnyAddress | None = IPvAnyAddress("192.168.1.254") # type: ignore
    dome_controller_ip_addr: IPvAnyAddress | None = None
    gps_port: str | None = Field("/dev/ttyUSB0")
    detector_overvoltage: int | None = None
    domes: list[ObsDomeConfig]

    model_config = ConfigDict(extra='allow')

    @model_validator(mode='after')
    def validate_wps_extras(self) -> ObsConfigValidator:
        """Ensures that dynamic 'wps-' keys match the WpsConfig schema."""
        extra_data = self.model_extra or {}
        for key, val in extra_data.items():
            if key.startswith("wps"):
                try:
                    WpsConfig(**val)
                except Exception as e:
                    raise ValueError(f"Invalid format for '{key}': {e}") from e
            else:
                raise ValueError(f"Extra key '{key}' is not allowed unless it's a 'wps' unit.")
        return self

# -------------------------
# --- DAQ Config Models ---
# -------------------------

class DaqNodeValidator(BaseStrictModel):
    username: str
    data_dir: str
    ip_addr: IPvAnyAddress
    module_ids: str | list[int]
    bindhost: str | None = Field("0.0.0.0")

    @field_validator('module_ids', mode='after')
    def validate_module_range(cls, v):
        if isinstance(v, list):
            # Module ids must be non-negative
            if not all(mid >= 0 for mid in v):
                raise ValueError(f"Invalid module IDs ({v}): Module ids must be non-negative")
            elif len(set(v)) != len(v):
                raise ValueError(f"Invalid module IDs ({v}): Module IDs must be unique if provided as a list of integers")
            elif len(v) == 0:
                raise ValueError(f"Invalid module IDs ({v}): Module IDs must be non-empty")
            else:
                return v
        elif isinstance(v, str):
            if re.match(r'^\d+\-\d+$', v):
                # print(v)
                start, end = map(int, v.split('-'))
                if start > end:
                    raise ValueError(f"Start module ID ({start}) must be <= End module ID ({end})")
                return v
            elif re.match(r'^(\d+)(, ?\d+)*$', v):
                # print(v)
                module_ids = list(map(int, v.split(',')))
                assert len(module_ids) == len(set(module_ids)), "module_ids in list format must be unique"
                return v
            elif re.match(r'^\[\d+\]$', v):
                return v
            else:
                raise ValueError("module_ids must be in the format 'start-end' (e.g., '0-127') OR '<module_id_A>, <module_id_B>, ..., <module_id_N>'")
        else:
            raise ValueError(f"Unexpected type for 'module_ids': '{type(v)=}'")


class DaqConfigValidator(BaseStrictModel):
    comment: str | None = None
    head_node_data_dir: str
    head_node_ip_addr: IPvAnyAddress
    head_node_container: bool = Field(False)
    daq_nodes: list[DaqNodeValidator]

    @model_validator(mode='after')
    def check_head_node_data_dir_match(self) -> DaqConfigValidator:
        # If the head node and the DAQ node are the same machine, data_dir must match.
        head_ip = str(self.head_node_ip_addr)
        for node in self.daq_nodes:
            if str(node.ip_addr) == head_ip:
                if node.data_dir != self.head_node_data_dir:
                    raise ValueError(
                        f"DAQ Node IP ({node.ip_addr}) matches head node, but "
                        f"data_dir '{node.data_dir}' does not match "
                        f"head_node_data_dir '{self.head_node_data_dir}'."
                    )
        return self


# -----------------------------
# --- Network Config Models ---
# -----------------------------

class PortForwarding(BaseStrictModel):
    status: bool
    gw_ip: IPvAnyAddress
    reboot_port: list[int | None] | None = Field(None)
    cmd_port: list[int | None] | None = Field(None)
    port: int | None = None                              # SSH forwarded port (legacy)
    grpc_port: int | None = Field(None, ge=1, le=65535)  # gRPC forwarded port

class NetworkModule(BaseStrictModel):
    ip_addr: IPvAnyAddress
    port_forwarding: PortForwarding

class NetworkDaqNode(BaseStrictModel):
    ip_addr: IPvAnyAddress
    port_forwarding: PortForwarding

class NetworkConfigValidator(BaseStrictModel):
    modules: list[NetworkModule]
    daq_nodes: list[NetworkDaqNode]

# ----------------------------
# --- Daemon Config Models ---
# ----------------------------

class Daemons(BaseModel):
    model_config = ConfigDict(extra='allow') # Allow dynamic casper_xx keys


class DaemonConfigValidator(BaseStrictModel):
    daemons: Daemons
    permanent_daemons: Daemons

# ------------------------------
# --- Firmware Config Models ---
# ------------------------------

class FirmwareConfigValidator(BaseModel):
    model_config = ConfigDict(extra='allow') # Allow 'qfp', 'bga', or future hardware variants

# --------------------------------
# --- Quabo UIDs Config Models ---
# --------------------------------

class QuaboUidEntry(BaseStrictModel):
    uid: str = Field(..., description="Hex string of the Quabo UID. Empty string if offline.")

class QuaboUidModule(BaseStrictModel):
    quabos: list[QuaboUidEntry] = Field(..., min_length=4, max_length=4)

    @field_validator('quabos')
    def ensure_four_quabos(cls, v):
        if len(v) != 4:
            raise ValueError(f"A module must specify exactly 4 quabos, found {len(v)}.")
        return v

class QuaboUidDome(BaseStrictModel):
    modules: list[QuaboUidModule]

class QuaboUidsValidator(BaseStrictModel):
    domes: list[QuaboUidDome]
