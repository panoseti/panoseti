"""
Pydantic models for validating data_config.json and interleave configuration.
Fail-fast schema checking to prevent observatory data corruption.
"""

import logging
import json
from rich.console import Console
from rich.pretty import pprint

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator

console = Console()

class AnyTriggerConfig(BaseModel):
    group_ph_frames: int = Field(0, description="If set to 1, hashpipe will group 4 packets from 4 quabos.")

class PulseHeightMode(BaseModel):
    pe_threshold: int = Field(..., description="Pulse height threshold in photoelectrons")
    any_trigger: Optional[AnyTriggerConfig] = None
    two_pixel_trigger: int = Field(0, description="If set to 1, 2 pixel trigger mode will be enabled.")
    three_pixel_trigger: int = Field(0, description="If set to 1, 3 pixel trigger mode will be enabled.")

class ImageMode(BaseModel):
    integration_time_usec: int = Field(..., description="Must divide 1000000")
    pe_threshold: int = Field(..., description="Image mode threshold in photoelectrons")
    quabo_sample_size: int = Field(..., description="Size of the sample")
    # nsum: int = Field(..., description="Sum this many images per output frame")
    quabo_num: Optional[int] = Field(None, description="Omit for all 4 quabos")

class InterleaveState(BaseModel):
    state_name: str = Field(..., description="A descriptive string for logging (e.g., 'Astrometry_Movie_Mode').")
    duration_seconds: float = Field(..., gt=0.0, description="Time in seconds to stay in this mode before switching.")
    movie_mode_config: Optional[str] = Field(None, description="The string key of the image mode to use. null to disable.")
    pulse_height_mode_config: Optional[str] = Field(None, description="The string key of the PH mode to use. null to disable.")

    @model_validator(mode='after')
    def check_at_least_one_mode(self) -> 'InterleaveState':
        if not self.movie_mode_config and not self.pulse_height_mode_config:
            raise ValueError(
                f"Configuration Error: State '{self.state_name}' must have at least one valid mode (movie or pulse_height). Both cannot be null.")
        return self

class InterleaveConfig(BaseModel):
    enable: bool = Field(False, description="If false or missing, interleaving is completely ignored.")
    states: List[InterleaveState] = Field([], description="A list indicating the switching order.")

class DataConfigValidator(BaseModel):
    """Validates the entire data_config.json file, including dynamic suffixed keys."""
    run_type: str = Field(..., description="science, engineering, or calibration.")
    detector_overvoltage: int = Field(..., description="Over voltage used for the observation. For now, we only have calibration data for 2V and 3V.")
    max_file_size_mb: int = Field(..., description="Start a new output file when this size is exceeded.")

    # Base modes (implicit id=0)
    image: Optional[ImageMode] = None
    pulse_height: Optional[PulseHeightMode] = None

    model_config = {"extra": "allow"}

    @model_validator(mode='after')
    def validate_interleave_and_exclusions(self) -> 'DataConfigValidator':
        extra_data = self.model_extra or {}

        ph_configs: Dict[str, PulseHeightMode] = {}
        img_configs: Dict[str, ImageMode] = {}

        if self.pulse_height:
            ph_configs["pulse_height"] = self.pulse_height
        if self.image:
            img_configs["image"] = self.image

        # Parse dynamically suffixed keys
        for key, val in extra_data.items():
            if key.startswith("pulse_height_"):
                ph_configs[key] = PulseHeightMode(**val)
            elif key.startswith("image_"):
                img_configs[key] = ImageMode(**val)

        # Extract interleave config
        interleave_data = extra_data.get("interleave")
        if not interleave_data or not interleave_data.get("enable", False):
            # If interleaving is disabled, we still must check the default mutual exclusion
            if self.image and self.pulse_height:
                if self.pulse_height.two_pixel_trigger > 0 or self.pulse_height.three_pixel_trigger > 0:
                     raise ValueError("MUTUAL EXCLUSION VIOLATION in default config: Cannot enable image mode while 2-pixel or 3-pixel triggers are enabled.")
            return self

        interleave = InterleaveConfig(**interleave_data)

        # Validate states and mutual exclusions
        for state in interleave.states:
            if state.movie_mode_config and state.movie_mode_config not in img_configs:
                raise ValueError(f"State '{state.state_name}' references missing movie_mode_config '{state.movie_mode_config}'")
            if state.pulse_height_mode_config and state.pulse_height_mode_config not in ph_configs:
                raise ValueError(f"State '{state.state_name}' references missing pulse_height_mode_config '{state.pulse_height_mode_config}'")

            # Check Mutual Exclusion Property: 2+ or 3+ pix trigger CANNOT exist with image mode
            if state.movie_mode_config and state.pulse_height_mode_config:
                ph_mode = ph_configs[state.pulse_height_mode_config]
                if ph_mode.two_pixel_trigger > 0 or ph_mode.three_pixel_trigger > 0:
                    raise ValueError(
                        f"MUTUAL EXCLUSION VIOLATION in state '{state.state_name}': "
                        f"Cannot enable image mode ('{state.movie_mode_config}') while 2-pixel or 3-pixel "
                        f"triggers are enabled in ('{state.pulse_height_mode_config}')."
                    )
        return self

def load_and_validate_data_config(filepath: str, logger: logging.Logger) -> dict:
    """Loads JSON, runs Pydantic validation, and returns the raw dict if successful."""
    with open(filepath, 'r') as f:
        raw_data = json.load(f)

    logger.info(f"Validating config: {filepath}")
    validated = DataConfigValidator(**raw_data)

    logger.info("[bold green]Configuration successfully validated![/bold green]", extra={"markup": True})
    console.print("Validated Configuration Tree:")
    pprint(validated.model_dump(), expand_all=True)

    return raw_data