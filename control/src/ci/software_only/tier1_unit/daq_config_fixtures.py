"""
daq_config_fixtures.py — Shared DataConfig dictionaries for unit and fleet tests.
"""

from __future__ import annotations

# 1. Image only (8-bit)
IMAGE_8BIT_ONLY = {
    "run_type": "eng-img8",
    "image": {
        "integration_time_usec": 5000,
        "pe_threshold": 3.0,
        "quabo_sample_size": 8
    }
}

# 2. Image only (16-bit)
IMAGE_16BIT_ONLY = {
    "run_type": "eng-img16",
    "image": {
        "integration_time_usec": 20,
        "pe_threshold": 1.0,
        "quabo_sample_size": 16
    }
}

# 3. Pulse height only (any_trigger off)
PH_ONLY = {
    "run_type": "eng-ph",
    "pulse_height": {
        "pe_threshold": 20.5
    }
}

# 4. Pulse height with grouping (any_trigger on)
PH_GROUPING = {
    "run_type": "eng-ph-group",
    "pulse_height": {
        "pe_threshold": 2.5,
        "any_trigger": {
            "group_ph_frames": 1
        }
    }
}

# 5. Dual mode (Image 16 + PH grouping)
DUAL_MODE = {
    "run_type": "eng-dual",
    "image": {
        "integration_time_usec": 10000,
        "pe_threshold": 5.0,
        "quabo_sample_size": 16
    },
    "pulse_height": {
        "pe_threshold": 3.0,
        "any_trigger": {
            "group_ph_frames": 1
        }
    }
}

# 6. Flash and Stim params
TEST_SIGNALS = {
    "run_type": "eng-test",
    "image": {
        "integration_time_usec": 1000,
        "pe_threshold": 3.0,
        "quabo_sample_size": 8
    },
    "flash_params": {
        "rate": 3,
        "level": 12,
        "width": 5
    },
    "stim_params": {
        "rate": 2,
        "level": 255,
        "mask": [True, True, False, False]
    }
}

# 7. Interleaving mode (complex case)
INTERLEAVED_CONFIG = {
    "run_type": "eng-interleave",
    "image": {
        "integration_time_usec": 10000,
        "pe_threshold": 3,
        "quabo_sample_size": 16
    },
    "image_8bit": {
        "integration_time_usec": 5000,
        "pe_threshold": 20.5,
        "quabo_sample_size": 8
    },
    "pulse_height": {
        "pe_threshold": 20.5,
        "any_trigger": {
            "group_ph_frames": 1
        }
    },
    "interleave": {
        "enable": True,
        "states": [
            {
                "state_name": "dual-mode",
                "duration_seconds": 1.0,
                "movie_mode_config": "image_8bit",
                "pulse_height_mode_config": "pulse_height"
            }
        ]
    }
}

VALID_CONFIGS = [
    IMAGE_8BIT_ONLY,
    IMAGE_16BIT_ONLY,
    PH_ONLY,
    PH_GROUPING,
    DUAL_MODE,
    TEST_SIGNALS,
    INTERLEAVED_CONFIG,
]

INVALID_CONFIGS = [
    # 2-pixel trigger with image mode (Hardware constraint)
    {
        "run_type": "fail-hw",
        "image": {
            "integration_time_usec": 10000,
            "pe_threshold": 3,
            "quabo_sample_size": 16
        },
        "pulse_height": {
            "pe_threshold": 3,
            "two_pixel_trigger": 1
        }
    },
    # Missing required fields in image mode
    {
        "run_type": "fail-missing",
        "image": {
            "integration_time_usec": 10000
        }
    },
    # Invalid integration time (not a divisor of 1,000,000)
    {
        "run_type": "fail-divisor",
        "image": {
            "integration_time_usec": 33,
            "pe_threshold": 1.0,
            "quabo_sample_size": 8
        }
    }
]
