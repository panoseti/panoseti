"""
Pydantic models for validating <type>_config.json files
"""

import logging
import json
from rich.console import Console
from rich.pretty import pprint

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator

# rich console
console = Console()

class AnyTriggerConfig(BaseModel):
    group_ph_frames: int = 0


class PulseHeightMode(BaseModel):
    pe_threshold: int
    any_trigger: Optional[AnyTriggerConfig] = None
    two_pixel_trigger: int = 0
    three_pixel_trigger: int = 0


class ImageMode(BaseModel):
    integration_time_usec: int
    pe_threshold: int
    quabo_sample_size: int
    nsum: int
    quabo_num: Optional[int] = None


class InterleaveState(BaseModel):
    state_name: str
    duration_seconds: float = Field(..., gt=0.0)
    movie_mode_config: Optional[str]
    pulse_height_mode_config: Optional[str]

    @model_validator(mode='after')
    def check_at_least_one_mode(self) -> 'InterleaveState':
        if not self.movie_mode_config and not self.pulse_height_mode_config:
            raise ValueError(
                f"State '{self.state_name}' must have at least one valid mode (movie or pulse_height). Both cannot be null.")
        return self


class InterleaveConfig(BaseModel):
    enable: bool = False
    states: List[InterleaveState] = []


class DataConfigValidator(BaseModel):
    """
    Validates the entire data_config.json file, including dynamic suffixed keys.
    """
    run_type: str
    detector_overvoltage: int
    max_file_size_mb: int

    # We use model_extra to capture dynamic keys like image_1, pulse_height_2
    model_config = {"extra": "allow"}

    @model_validator(mode='after')
    def validate_interleave_and_exclusions(self) -> 'DataConfigValidator':
        extra_data = self.model_extra or {}

        # 1. Parse dynamic mode configs
        ph_configs: Dict[str, PulseHeightMode] = {}
        img_configs: Dict[str, ImageMode] = {}

        # Manually parse standard modes if they exist in extra (they usually will if dumped flat)
        for key, val in extra_data.items():
            if key.startswith("pulse_height"):
                ph_configs[key] = PulseHeightMode(**val)
            elif key.startswith("image"):
                img_configs[key] = ImageMode(**val)

        # 2. Extract interleave config
        interleave_data = extra_data.get("interleave")
        if not interleave_data or not interleave_data.get("enable", False):
            return self  # Interleaving disabled, skip deep validation

        interleave = InterleaveConfig(**interleave_data)

        # 3. Validate states and mutual exclusions
        for state in interleave.states:
            # Check if referenced keys exist
            if state.movie_mode_config and state.movie_mode_config not in img_configs:
                raise ValueError(
                    f"State '{state.state_name}' references missing movie_mode_config '{state.movie_mode_config}'")
            if state.pulse_height_mode_config and state.pulse_height_mode_config not in ph_configs:
                raise ValueError(
                    f"State '{state.state_name}' references missing pulse_height_mode_config '{state.pulse_height_mode_config}'")

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