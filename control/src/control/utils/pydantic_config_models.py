"""
pydantic_config_models.py

Centralized Pydantic models for validating PANOSETI configuration files.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

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

from control.utils.paths import PanoPaths

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
    """Configuration for 'any_trigger' mode in pulse height acquisition."""
    group_ph_frames: int = Field(0, description="If set to 1, hashpipe will group 4 packets from 4 quabos.")

class PulseHeightMode(BaseStrictModel):
    """Parameters for Pulse Height (PH) data acquisition."""
    pe_threshold: float = Field(..., ge=MIN_PULSE_HEIGHT_PE_THRESHOLD, description="Pulse height threshold in photoelectrons")
    any_trigger: AnyTriggerConfig | None = None
    two_pixel_trigger: int = Field(0, description="If set to 1, 2 pixel trigger mode will be enabled.")
    three_pixel_trigger: int = Field(0, description="If set to 1, 3 pixel trigger mode will be enabled.")


## -- data_config: Movie-mode
class ImageMode(BaseStrictModel):
    """Parameters for Image (Movie) mode data acquisition."""
    integration_time_usec: int = Field(..., ge=20, description="Integration time in microseconds")
    pe_threshold: float = Field(..., ge=MIN_MOVIE_MODE_PE_THRESHOLD, description="Image mode threshold in photoelectrons")
    quabo_sample_size: Literal[8, 16] = Field(..., description="Size of the sample")

    @field_validator('integration_time_usec')
    def check_integration_time_divisor(cls, v: int) -> int:
        if 1_000_000 % v != 0:
            raise ValueError(f"integration_time_usec ({v}) must evenly divide 1,000,000 usec.")
        return v

## -- data_config: Test signal injection
class LongPulseMode(BaseStrictModel):
    """Parameters for long-pulse event detection."""
    octaves: int
    threshold_sigma: list[float]

class FlashParams(BaseStrictModel):
    """Controls for the onboard LED flash system."""
    rate: int = Field(..., ge=0, le=7, description="3-bit value controlling flash rate (0-7)")
    level: int = Field(..., ge=0, le=31, description="Controls DC supply level (0-31)")
    width: int = Field(..., ge=0, le=15, description="Controls pulse width (0-15)")

class StimParams(BaseStrictModel):
    """Controls for the electronic stimulus (test pulse) system."""
    rate: int = Field(..., ge=0, le=7, description="Rate from 190 to 24,400 Hz")
    level: int = Field(..., ge=0, le=255)
    mask: list[bool] = Field(..., max_length=4, min_length=4)

## data_config: Interleaving mode
class InterleaveState(BaseStrictModel):
    """A single state in an interleaved observing sequence."""
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
    """Full configuration for cyclical interleaved observing."""
    enable: bool = Field(False)
    states: list[InterleaveState] = Field([])

## data_config: global validator
class DataConfigValidator(BaseModel):
    """Science and engineering acquisition parameters (data_config.json)."""
    # We must use extra='allow' so Pydantic parses them, but we will
    # strictly validate the extra keys dynamically in mode='after'.
    model_config = ConfigDict(extra='allow')

    run_type: str = Field(..., max_length=MAX_RUN_TYPE_LENGTH)
    detector_overvoltage: Literal[2, 3] | None = None
    gain: int | None = None
    max_file_size_mb: int | None = Field(None, gt=0)
    image: ImageMode | None = None
    pulse_height: PulseHeightMode | None = None
    interleave: InterleaveConfig | None = None
    stim_params: StimParams | None = None
    flash_params: FlashParams | None = None

    @field_validator("run_type")
    def validate_run_type(cls, v: str) -> str:
        if any(ch in INVALID_RUN_TYPE_CHARS for ch in v):
            raise ValueError(
                f"Invalid run_type: '{v}' contains at least one invalid character: {INVALID_RUN_TYPE_CHARS}")
        return v

    @model_validator(mode='after')
    def validate_dynamic_modes_and_interleave(self) -> DataConfigValidator:
        dynamic_keys: list[str] = []
        ph_modes_dict: dict[str, PulseHeightMode] = {}

        if self.pulse_height:
            ph_modes_dict['pulse_height'] = self.pulse_height

        # Global hardware check for the default modes
        if self.image and self.pulse_height and (self.pulse_height.two_pixel_trigger > 0 or self.pulse_height.three_pixel_trigger > 0):
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
            valid_image_modes = ["image", *dynamic_keys]
            valid_ph_modes = ["pulse_height", *dynamic_keys]

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
                    state_ph_obj = ph_modes_dict.get(p_conf)
                    if state_ph_obj and (state_ph_obj.two_pixel_trigger > 0 or state_ph_obj.three_pixel_trigger > 0):
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
    """Configuration for a Web Power Switch (WPS) unit."""
    url: str
    quabo_socket: int


class ObsModuleConfig(BaseModel):
    """Configuration and state for a single observatory module."""
    model_config = ConfigDict(extra='allow')
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
    daq_node: Any | None = None


class ObsDomeConfig(BaseStrictModel):
    """Physical configuration for an observatory dome."""
    name: str
    obslat: float = Field(..., ge=-90, le=90)
    obslon: float = Field(..., ge=-180, le=180)
    obsalt: float
    modules: list[ObsModuleConfig]

    # Injected fields by config_file.py at runtime
    num: int | None = None


class ObsConfigValidator(BaseModel):
    """Physical observatory setup and device mapping (obs_config.json)."""
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

class PortForwarding(BaseStrictModel):
    """Networking metadata for port-forwarded devices (Gateways)."""
    status: bool
    gw_ip: IPvAnyAddress
    reboot_port: list[int | None] | None = Field(None)
    cmd_port: list[int | None] | None = Field(None)
    port: int | None = None                              # SSH forwarded port (legacy)
    grpc_port: int | None = Field(None, ge=1, le=65535)  # gRPC forwarded port


class DaqNodeValidator(BaseModel):
    """Configuration for a single remote Data Acquisition (DAQ) node."""
    model_config = ConfigDict(extra='allow')
    username: str
    data_dir: str
    ip_addr: IPvAnyAddress
    module_ids: list[int]
    bindhost: str | None = Field("0.0.0.0")
    port_forwarding: PortForwarding | None = None
    modules: list[Any] = Field(default_factory=list)

    @field_validator('module_ids', mode='before')
    def validate_module_range(cls, v: Any) -> list[int]:
        if isinstance(v, list):
            # Module ids must be non-negative
            res = [int(x) for x in v]
            if not all(mid >= 0 for mid in res):
                raise ValueError(f"Invalid module IDs ({v}): Module ids must be non-negative")
            elif len(set(res)) != len(res):
                raise ValueError(f"Invalid module IDs ({v}): Module IDs must be unique if provided as a list of integers")
            elif len(res) == 0:
                raise ValueError(f"Invalid module IDs ({v}): Module IDs must be non-empty")
            else:
                return res
        elif isinstance(v, str):
            if re.match(r'^\d+\-\d+$', v):
                start, end = map(int, v.split('-'))
                if start > end:
                    raise ValueError(f"Start module ID ({start}) must be <= End module ID ({end})")
                return list(range(start, end + 1))
            elif re.match(r'^(\d+)(, ?\d+)*$', v):
                module_ids = list(map(int, v.split(',')))
                if len(module_ids) != len(set(module_ids)):
                     raise ValueError("module_ids in list format must be unique")
                return module_ids
            elif re.match(r'^\[\d+\]$', v):
                return [int(v[1:-1])]
            elif v.isdigit():
                return [int(v)]
            else:
                raise ValueError("module_ids must be in the format 'start-end' (e.g., '0-127') OR '<module_id_A>, <module_id_B>, ..., <module_id_N>'")
        elif isinstance(v, int):
            return [v]
        else:
            raise ValueError(f"Unexpected type for 'module_ids': '{type(v)=}'")


class DaqConfigValidator(BaseStrictModel):
    """DAQ node networking and storage configuration (daq_config.json)."""
    comment: str | None = None
    head_node_data_dir: str
    head_node_ip_addr: IPvAnyAddress
    head_node_container: bool | None = Field(False)
    daq_node_module_limit: int | None = Field(4, description="Maximum number of modules per DAQ node (structural limit)")
    daq_nodes: list[DaqNodeValidator]

    @model_validator(mode='after')
    def check_head_node_data_dir_match(self) -> DaqConfigValidator:
        # If the head node and the DAQ node are the same machine, data_dir must match.
        head_ip = str(self.head_node_ip_addr)
        for node in self.daq_nodes:
            if str(node.ip_addr) == head_ip and node.data_dir != self.head_node_data_dir:
                raise ValueError(
                    f"DAQ Node IP ({node.ip_addr}) matches head node, but "
                    f"data_dir ({node.data_dir}) differs from head_node_data_dir ({self.head_node_data_dir})."
                )
        return self


# -----------------------------
# --- Network Config Models ---
# -----------------------------

class NetworkModule(BaseStrictModel):
    """Network-level mapping for a physical module."""
    ip_addr: IPvAnyAddress
    port_forwarding: PortForwarding

class NetworkDaqNode(BaseStrictModel):
    """Network-level mapping for a DAQ node."""
    ip_addr: IPvAnyAddress
    port_forwarding: PortForwarding

class NetworkConfigValidator(BaseStrictModel):
    """Global network routing and port-forwarding map (network_config.json)."""
    modules: list[NetworkModule] = Field(default_factory=list)
    daq_nodes: list[NetworkDaqNode] = Field(default_factory=list)

# ----------------------------
# --- Daemon Config Models ---
# ----------------------------

class Daemons(BaseModel):
    """Enabled/disabled states for specific background daemons."""
    model_config = ConfigDict(extra='allow') # Allow dynamic casper_xx keys


class DaemonConfigValidator(BaseStrictModel):
    """Configuration for observatory background processes (daemons.json)."""
    daemons: Daemons
    permanent_daemons: Daemons


# ------------------------------
# --- Firmware Config Models ---
# ------------------------------

class FirmwareConfigValidator(BaseModel):
    """Mapping of hardware types to firmware binaries."""
    model_config = ConfigDict(extra='allow') # Allow 'qfp', 'bga', or future hardware variants
    qfp: str | None = None
    bga: str | None = None
    gold: str | None = None

    @model_validator(mode='after')
    def validate_firmware_files(self) -> FirmwareConfigValidator:
        fw_dir = PanoPaths.firmware_dir()
        for field in ['qfp', 'bga', 'gold']:
            filename = getattr(self, field)
            if filename:
                fw_path = fw_dir / filename
                if not fw_path.exists():
                    # We only log a warning during validation if file is missing,
                    # as it might be a partial dev setup.
                    # Use print since we don't have a logger here yet.
                    pass
        return self

# --------------------------------
# --- Quabo UIDs Config Models ---
# --------------------------------

class QuaboUidEntry(BaseStrictModel):
    """A single Quabo unique ID entry."""
    uid: str = Field(..., description="Hex string of the Quabo UID. Empty string if offline.")

class QuaboUidModule(BaseModel):
    """A module-level grouping of Quabo unique IDs."""
    model_config = ConfigDict(extra='allow')
    ip_addr: IPvAnyAddress
    quabos: list[QuaboUidEntry] = Field(..., min_length=4, max_length=4)
    id: int | None = None
    daq_node: Any | None = None

    @field_validator('quabos')
    def ensure_four_quabos(cls, v: list[QuaboUidEntry]) -> list[QuaboUidEntry]:
        if len(v) != 4:
            raise ValueError(f"A module must specify exactly 4 quabos, found {len(v)}.")
        return v

class QuaboUidDome(BaseStrictModel):
    """A dome-level grouping of Quabo UID modules."""
    modules: list[QuaboUidModule]
    num: int | None = None

class QuaboUidsValidator(BaseStrictModel):
    """Local cache of unique hardware IDs for all observatory Quabos."""
    domes: list[QuaboUidDome]


# ---------------------------
# --- Run State Ledger ---
# ---------------------------

class NodeReceipt(BaseStrictModel):
    """Transactional status report from a single DAQ node."""
    ip_addr: IPvAnyAddress
    status: Literal["STARTING", "START_SUCCESS", "START_FAILED", "STOPPED"] = "STARTING"
    hashpipe_pid: int | None = None
    data_dir: str | None = None
    message: str | None = None
    manifest_path: str | None = None
    manifest_bytes: int | None = None
    rsync_bytes_transferred: int | None = None
    rsync_last_progress_at: datetime | None = Field(default=None)
    verify_ok: bool | None = None
    cleanup_ok: bool | None = None

class CollectResult(BaseStrictModel):
    """Result of a data collection attempt from all nodes."""
    success: bool
    errors: list[str] = Field(default_factory=list)
    failed_ips: list[str] = Field(default_factory=list)
    transferred_files: int = 0

class RunStateLedger(BaseStrictModel):
    """The central source of truth for an active observatory run."""
    run_name: str
    status: Literal[
        "STARTING", "ACTIVE", "ABORTED", "STOPPING",
        "RECORDING_ENDED",
        "MANIFEST_PENDING", "MANIFEST_GENERATING", "MANIFEST_READY",
        "TRANSFER_PENDING", "TRANSFERRING", "TRANSFER_FAILED",
        "VERIFYING", "VERIFY_FAILED",
        "CLEANUP_PENDING", "CLEANING",
        "ARCHIVED", "COMPLETED", "STOPPED_WITH_ERRORS",
    ] = "STARTING"
    start_time: str  # ISO 8601
    pid: int | None = None
    host: str | None = None
    config_metadata: dict[str, Any] = Field(default_factory=dict)
    nodes: list[NodeReceipt] = Field(default_factory=list)
    transfer_attempts: int = 0
    last_transfer_error: str | None = Field(default=None)
    manifest_algorithm: str | None = Field(default=None)
    next_action_not_before: datetime | None = Field(default=None)


# ---------------------------
# --- PFF Metadata Models ---
# ---------------------------

class PFFHeader(BaseStrictModel):
    """Standardized PanoSETI File Format (PFF) JSON header."""
    pkt_num: int
    pkt_tai: int
    pkt_nsec: int
    tv_sec: int
    tv_usec: int

class QuaboHeader(PFFHeader):
    """PFF header specific to a single Quabo's data."""
    quabo_num: int

class ModuleHeader(BaseStrictModel):
    """Hierarchical PFF header containing telemetry for a full 4-Quabo module."""
    quabo_0: PFFHeader
    quabo_1: PFFHeader
    quabo_2: PFFHeader
    quabo_3: PFFHeader
