# Interleaved Mode Observation

The PANOSETI system supports rapid, automated switching between different configuration states (e.g., alternating between a primary science trigger mode and a periodic movie-mode for astrometry) during an active run.

## Using Interleave Modes
Interleaving is an overlay on top of standard observing. You must start a standard run before interleaving can begin.

1. **Validate Configs:** Calls Pydantic models to check if all configuration files specify valid observing states and adhere to configuration schemas.
   * `python config.py --validate` reports OK or specific errors for each configuration file and their absolute paths.
   * `python config.py --validate debug` additionally prints the contents of configuration files.
   * `python config.py --validate network` validates configuration files and checks network connectivity to observatory devices.
2. **Start Observation:** `python start.py` (Initializes the run using the default `image` and `pulse_height` keys).
3. **Begin Interleaving:** `python config.py --start-interleave` (Runs the scheduler in the background).
4. **Stop Interleaving:** `python config.py --stop-interleave` (Gracefully stops the scheduler and returns the Quabos to the default `image` and `pulse_height` state).
5. **Stop Observation:** `python stop.py`
   6. *(Note: Running `stop.py` while interleaving is active will automatically terminate the interleaver and stop data flow).*

## Defining Interleave States
To define alternative modes in `data_config.json`, you must use keys that begin with the prefix `image_` or `pulse_height_`. The suffix can be any descriptive name (e.g., `pulse_height_DUAL`, `image_astrometry`).

* **Note:** The root keys `image` and `pulse_height` are strictly reserved for the **default** operating state.

```json
{
    "pulse_height": { ... }, 
    "image": { ... },
    "pulse_height_uhe": { ... }, 
    "image_8bit": { ... },
    "interleave": {
        "enable": true,
        "states": [
            {
                "state_name": "uhe-science-2pix-ph1024",
                "duration_seconds": 58.0,
                "movie_mode_config": null,
                "pulse_height_mode_config": "pulse_height_uhe"
            },
            {
                "state_name": "astrometry-dual-mode",
                "duration_seconds": 2.0,
                "movie_mode_config": "image_8bit",
                "pulse_height_mode_config": "pulse_height"
            },
            {
                "state_name": "astrometry-img8-only",
                "duration_seconds": 2.0,
                "movie_mode_config": "image_8bit",
                "pulse_height_mode_config": null
            }
        ]
    }
}

```

### Interleave Configuration (Extends `data_config.json`)

To support rapid mode switching during a single observation, `data_config.json` now supports optional "interleaved" mode definitions and scheduling.

Additional modes can be defined by appending an underscore and an ID to the standard mode keys (e.g., `image_1`, `pulse_height_2`). These are ignored by default unless explicitly scheduled in the `interleave` block.

* **interleave**: Contains the schedule for mode switching. (Overrides the initial config during the interleaving loop).
* **enable**: `true` or `false`. If `false` or missing, interleaving is completely ignored, and the system uses only the default `image` and `pulse_height` configs (implicit id=0).
* **states**: A list of state definition objects indicating the switching order. The script will loop through this array infinitely.
  * **state_name**: A descriptive string for logging (e.g., "dual-mode-img8-ph1024").
  * **duration_seconds**: Minimum time in seconds to stay in this mode before executing the next switch.
  * **movie_mode_config**: The string key of the image mode to use (e.g., `"image_8bit"` or `"image"`). Set to `null` to disable image mode for this state.
  * **pulse_height_mode_config**: The string key of the pulse height mode to use (e.g., `"pulse_height_2pix-ph1024"` or `"pulse_height"`). Set to `null` to disable pulse height mode for this state.


**Note:** A given interleaving state cannot have both `movie_mode_config` and `pulse_height_mode_config` set to `null`. Furthermore, due to hardware/firmware constraints, if a pulse height mode enables `two_pixel_trigger` or `three_pixel_trigger` (> 0), movie-mode imaging *cannot* be enabled in the same interval state. These rules are automatically to `data_config.json` by Pydantic validation rules. 

