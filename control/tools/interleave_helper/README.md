### Interleaved Mode Observation

The PANOSETI system supports rapid, automated switching between different configuration states (e.g., alternating between a primary science trigger mode and a periodic movie-mode for astrometry) during an active run.

**Defining Interleave States:**
To define alternative modes in `data_config.json`, you must use keys that begin with the prefix `image_` or `pulse_height_`. The suffix can be any descriptive name (e.g., `pulse_height_DUAL`, `image_astrometry`).

* **Note:** The root keys `image` and `pulse_height` are strictly reserved for the **default** operating state.

```json
{
    "pulse_height": { ... }, 
    "image": { ... },
    "pulse_height_gamma": { ... }, 
    "image_astrometry": { ... },
    "interleave": {
        "enable": true,
        "states": [
            {
                "state_name": "Gamma_Science",
                "duration_seconds": 58.0,
                "movie_mode_config": null,
                "pulse_height_mode_config": "pulse_height_gamma"
            },
            {
                "state_name": "Astrometry",
                "duration_seconds": 2.0,
                "movie_mode_config": "image_astrometry",
                "pulse_height_mode_config": "pulse_height"
            }
        ]
    }
}

```

**Usage Workflow:**
Interleaving is an overlay on top of standard observing. You must start a standard run before interleaving can begin.

1. **Validate Configs:** `python config.py --validate-configs` (Fails fast if your JSON is malformed or violates hardware rules).
2. **Start Observation:** `python start.py` (Initializes the run using the default `image` and `pulse_height` keys).
3. **Begin Interleaving:** `python config.py --start-interleave` (Runs the scheduler in the background).
4. **Stop Interleaving:** `python config.py --stop-interleave` (Gracefully stops the scheduler and returns the Quabos to the default `image` and `pulse_height` state).
5. **Stop Observation:** `python stop.py`

*(Note: Running `stop.py` while interleaving is active will automatically terminate the interleaver and stop data flow).*
