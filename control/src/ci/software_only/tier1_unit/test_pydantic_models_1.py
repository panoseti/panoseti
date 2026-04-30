# mypy: ignore-errors
"""
test_pydantic_models_1.py

Comprehensive unit tests for control/utils/pydantic_config_models.py.
Covers every field boundary, cross-field validator, and interleave constraint.
No hardware required.
"""



import pytest
from pydantic import ValidationError

from control.utils.pydantic_config_models import (
    AnyTriggerConfig,
    DataConfig,
    ImageMode,
    PulseHeightMode,
)

# ===========================================================================
# ImageMode
# ===========================================================================

class TestImageMode:
    def test_valid_image_mode(self) -> None:
        m = ImageMode(integration_time_usec=100_000, pe_threshold=1.0, quabo_sample_size=16)
        assert m.integration_time_usec == 100_000
        assert m.pe_threshold == 1.0

    @pytest.mark.parametrize("usec", [20, 100, 1000, 10_000, 100_000, 200_000, 500_000, 1_000_000])
    def test_valid_integration_times(self, usec) -> None:
        """All values that evenly divide 1,000,000 and are >= 20 are valid."""
        ImageMode(integration_time_usec=usec, pe_threshold=1.0, quabo_sample_size=16)

    @pytest.mark.parametrize("usec", [21, 30, 99_999, 333_333, 700_000])
    def test_invalid_integration_time_non_divisor(self, usec) -> None:
        """Values >= 20 that don't evenly divide 1,000,000 must be rejected."""
        with pytest.raises(ValidationError, match="must evenly divide"):
            ImageMode(integration_time_usec=usec, pe_threshold=1.0, quabo_sample_size=16)

    def test_integration_time_below_minimum(self) -> None:
        with pytest.raises(ValidationError):
            ImageMode(integration_time_usec=10, pe_threshold=1.0, quabo_sample_size=16)

    def test_pe_threshold_at_minimum(self) -> None:
        """pe_threshold == 1.0 is the minimum for image mode."""
        m = ImageMode(integration_time_usec=100_000, pe_threshold=1.0, quabo_sample_size=16)
        assert m.pe_threshold == 1.0

    def test_pe_threshold_below_minimum(self) -> None:
        with pytest.raises(ValidationError):
            ImageMode(integration_time_usec=100_000, pe_threshold=0.9, quabo_sample_size=16)

    @pytest.mark.parametrize("size", [8, 16])
    def test_valid_quabo_sample_sizes(self, size) -> None:
        ImageMode(integration_time_usec=100_000, pe_threshold=1.0, quabo_sample_size=size)

    def test_invalid_quabo_sample_size(self) -> None:
        with pytest.raises(ValidationError):
            ImageMode(integration_time_usec=100_000, pe_threshold=1.0, quabo_sample_size=4)  # type: ignore

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ImageMode(integration_time_usec=100_000, pe_threshold=1.0,
                      quabo_sample_size=16, unexpected_key=True)


# ===========================================================================
# PulseHeightMode
# ===========================================================================

class TestPulseHeightMode:
    def test_valid_pulse_height_mode(self) -> None:
        m = PulseHeightMode(pe_threshold=3.0, two_pixel_trigger=0, three_pixel_trigger=0)
        assert m.pe_threshold == 3.0
        assert m.two_pixel_trigger == 0
        assert m.three_pixel_trigger == 0

    def test_pe_threshold_at_minimum(self) -> None:
        PulseHeightMode(pe_threshold=2.0, two_pixel_trigger=0, three_pixel_trigger=0)

    def test_pe_threshold_below_minimum(self) -> None:
        with pytest.raises(ValidationError):
            PulseHeightMode(pe_threshold=1.9, two_pixel_trigger=0, three_pixel_trigger=0)

    def test_two_pixel_trigger(self) -> None:
        m = PulseHeightMode(pe_threshold=3.0, two_pixel_trigger=1, three_pixel_trigger=0)
        assert m.two_pixel_trigger == 1

    def test_three_pixel_trigger(self) -> None:
        m = PulseHeightMode(pe_threshold=3.0, two_pixel_trigger=0, three_pixel_trigger=1)
        assert m.three_pixel_trigger == 1

    def test_any_trigger_config(self) -> None:
        m = PulseHeightMode(
            pe_threshold=3.0,
            two_pixel_trigger=0,
            three_pixel_trigger=0,
            any_trigger=AnyTriggerConfig(group_ph_frames=1)
        )
        assert m.any_trigger is not None
        assert m.any_trigger.group_ph_frames == 1

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            PulseHeightMode(pe_threshold=3.0, two_pixel_trigger=0, three_pixel_trigger=0, unexpected=True) # type: ignore[call-arg]


# ===========================================================================
# DataConfig
# ===========================================================================

class TestDataConfig:
    def test_valid_minimal_data_config(self, minimal_data_config) -> None:
        v = DataConfig(**minimal_data_config)
        assert v.run_type == "sci"

    def test_run_type_max_length(self) -> None:
        """run_type must be <= 14 characters."""
        DataConfig(run_type="a" * 14)

    def test_run_type_too_long(self) -> None:
        with pytest.raises(ValidationError):
            DataConfig(run_type="a" * 15)

    @pytest.mark.parametrize("bad_char", [".", "_", " "])
    def test_run_type_invalid_chars(self, bad_char) -> None:
        with pytest.raises(ValidationError, match="invalid character"):
            DataConfig(run_type=f"sci{bad_char}run")

    def test_run_type_valid_alphanumeric(self) -> None:
        DataConfig(run_type="scicalib01")

    def test_detector_overvoltage_valid(self) -> None:
        """Only 2 and 3 are valid overvoltage values."""
        DataConfig(run_type="sci", detector_overvoltage=2)
        DataConfig(run_type="sci", detector_overvoltage=3)

    def test_detector_overvoltage_invalid(self) -> None:
        with pytest.raises(ValidationError):
            DataConfig(run_type="sci", detector_overvoltage=5)  # type: ignore

    def test_image_and_ph_without_trigger_ok(self) -> None:
        """Image + PH with no multi-pixel trigger is valid."""
        DataConfig(**{  # type: ignore
            "run_type": "sci",
            "image": {"integration_time_usec": 100_000, "pe_threshold": 1.0, "quabo_sample_size": 16},
            "pulse_height": {"pe_threshold": 3.0},
        })

    def test_image_and_ph_with_two_pixel_trigger_raises(self) -> None:
        """image + PH with two_pixel_trigger violates hardware constraint."""
        with pytest.raises(ValidationError, match="Hardware Constraint"):
            DataConfig(**{  # type: ignore
                "run_type": "sci",
                "image": {"integration_time_usec": 100_000, "pe_threshold": 1.0, "quabo_sample_size": 16},
                "pulse_height": {"pe_threshold": 3.0, "two_pixel_trigger": 1},
            })

    def test_image_and_ph_with_three_pixel_trigger_raises(self) -> None:
        with pytest.raises(ValidationError, match="Hardware Constraint"):
            DataConfig(**{  # type: ignore
                "run_type": "sci",
                "image": {"integration_time_usec": 100_000, "pe_threshold": 1.0, "quabo_sample_size": 16},
                "pulse_height": {"pe_threshold": 3.0, "three_pixel_trigger": 1},
            })

    def test_dynamic_image_key_valid(self) -> None:
        """Keys prefixed 'image_' must be valid ImageMode objects."""
        DataConfig(**{  # type: ignore
            "run_type": "sci",
            "image_8bit": {"integration_time_usec": 200_000, "pe_threshold": 1.0, "quabo_sample_size": 8},
        })

    def test_dynamic_image_key_invalid_fields(self) -> None:
        with pytest.raises(ValidationError, match="Invalid fields in dynamic mode"):
            DataConfig(**{  # type: ignore
                "run_type": "sci",
                "image_8bit": {"integration_time_usec": 7, "pe_threshold": 1.0, "quabo_sample_size": 8},
            })

    def test_dynamic_ph_key_valid(self) -> None:
        """Keys prefixed 'pulse_height_' must be valid PulseHeightMode objects."""
        DataConfig(**{  # type: ignore
            "run_type": "sci",
            "pulse_height_uhe": {"pe_threshold": 5.0},
        })

    def test_unrecognized_top_level_key_raises(self) -> None:
        """Keys that are neither image_* nor pulse_height_* must be rejected."""
        with pytest.raises(ValidationError, match="Unrecognized configuration key"):
            DataConfig(**{"run_type": "sci", "foobar": {"x": 1}})  # type: ignore

    def test_max_file_size_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            DataConfig(run_type="sci", max_file_size_mb=0)

    def test_max_file_size_valid(self) -> None:
        DataConfig(run_type="sci", max_file_size_mb=500)
