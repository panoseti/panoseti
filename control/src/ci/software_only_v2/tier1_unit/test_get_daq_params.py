"""
test_get_daq_params.py — Unit tests for get_daq_params logic.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from control.driver.quabo_driver import get_daq_params
from control.utils.pydantic_config_models import DataConfig
from ci.software_only_v2.tier1_unit.daq_config_fixtures import VALID_CONFIGS, INVALID_CONFIGS

@pytest.mark.tier1
@pytest.mark.parametrize("config_dict", VALID_CONFIGS)
def test_when_config_valid_then_get_daq_params_translates_correctly(config_dict: dict) -> None:
    # 1. Parse and validate DataConfig
    data_config = DataConfig(**config_dict)
    
    # 2. Translate to DAQ_PARAMS
    params = get_daq_params(data_config)
    
    # 3. Assertions based on config_dict structure
    if "image" in config_dict:
        assert params.do_image is True
        assert params.image_8bit == (config_dict["image"]["quabo_sample_size"] == 8)
        assert params.image_us == config_dict["image"]["integration_time_usec"] - 1
    else:
        assert params.do_image is False
        
    if "pulse_height" in config_dict:
        assert params.do_ph is True
        ph = config_dict["pulse_height"]
        if "any_trigger" in ph:
            assert params.do_any_trigger is True
            assert params.do_group_ph_frames == bool(ph["any_trigger"].get("group_ph_frames", 0))
    else:
        assert params.do_ph is False

    if "flash_params" in config_dict:
        assert params.do_flash is True
        fp = config_dict["flash_params"]
        assert params.flash_rate == fp["rate"]
        assert params.flash_level == fp["level"]
        assert params.flash_width == fp["width"]

    if "stim_params" in config_dict:
        assert params.do_stim is True
        sp = config_dict["stim_params"]
        assert params.stim_rate == sp["rate"]
        assert params.stim_level == sp["level"]

@pytest.mark.tier1
@pytest.mark.parametrize("config_dict", INVALID_CONFIGS)
def test_when_config_invalid_then_pydantic_raises_validation_error(config_dict: dict) -> None:
    with pytest.raises(ValidationError):
        DataConfig(**config_dict)
