### 1. Extended `data_config.json` Example

```json
{
    "run_type": "science",
    "detector_overvoltage": 3,
    "max_file_size_mb": 1000,
    "pulse_height": {
        "pe_threshold": 3,
        "any_trigger": {
            "group_ph_frames": 0
        },
        "two_pixel_trigger": 0,
        "three_pixel_trigger": 0
    },
    "pulse_height_1": {
        "pe_threshold": 3,
        "any_trigger": {
            "group_ph_frames": 0
        },
        "two_pixel_trigger": 1,
        "three_pixel_trigger": 0
    },
    "image_1": {
        "integration_time_usec": 20,
        "pe_threshold": 3,
        "quabo_sample_size": 8,
        "nsum": 64
    },
    "interleave": {
        "enable": true,
        "states": [
            {
                "state_name": "Trigger_Science_Mode",
                "duration_seconds": 58.0,
                "movie_mode_config": null,
                "pulse_height_mode_config": "pulse_height_1"
            },
            {
                "state_name": "Astrometry_Movie_Mode",
                "duration_seconds": 2.0,
                "movie_mode_config": "image_1",
                "pulse_height_mode_config": "pulse_height"
            }
        ]
    }
}

```

#### Interleaving Observation config (Optional)

To support rapid mode switching during a single observation (e.g., to capture intermittent movie-mode data for astrometry without sacrificing continuous multi-pixel trigger science data), `data_config.json` supports optional interleaved mode definitions and scheduling.

Additional modes can be defined by appending an underscore and an ID to the standard mode keys (e.g., `image_1`, `pulse_height_2`). These are ignored by default unless explicitly scheduled in the `interleave` block.

* **interleave**: Contains the schedule for mode switching. (Overrides the initial config during the interleaving loop).
* **enable**: `true` or `false`. If `false` or missing, interleaving is completely ignored, and the system uses only the default `image` and `pulse_height` configs (implicit id=0).
* **states**: A list of state definition objects indicating the switching order. The script will loop through this array infinitely.
* **state_name**: A descriptive string for logging (e.g., "Astrometry_Movie_Mode").
* **duration_seconds**: Time in seconds to stay in this mode before executing the next switch.
* **movie_mode_config**: The string key of the image mode to use (e.g., `"image_1"` or `"image"`). Set to `null` to disable image mode for this state.
* **pulse_height_mode_config**: The string key of the pulse height mode to use (e.g., `"pulse_height_1"` or `"pulse_height"`). Set to `null` to disable pulse height mode for this state.





**Note:** A given interleaving state cannot have both `movie_mode_config` and `pulse_height_mode_config` set to `null`. Furthermore, due to hardware/firmware constraints, if a pulse height mode enables `two_pixel_trigger` or `three_pixel_trigger` (> 0), movie-mode imaging *cannot* be enabled in the same interval state.