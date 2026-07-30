# Legacy DAQ-stop fallback

A vendored, minimal subset of the old (`mount-metadata` branch) control
software: just enough to stop an active recording -- hashpipe on every DAQ
node, and Quabo data flow -- **without depending on gRPC at all**. Use this
when the new stack's `daq_control` gRPC path is broken and `pseti stop`
doesn't work.

This is *not* a second way to run full observing sessions. It does one
job: get the observatory from "recording, new stack unresponsive" back to
a session-start-like state (DAQ stopped, quabos idle, hardware still
powered/configured) so an operator can bring up a session on either stack
afterward. Full old-lifecycle scripts (`session_start.py`, `start.py`,
calibration tooling, ...) were deliberately left out of scope -- see the
`origin/mount-metadata` branch if a complete parallel old-software
checkout is ever needed instead.

Verified against real hardware (UCB test rig): started a recording via
the new stack's `pseti admin attach headnode` → `pseti start`, confirmed
hashpipe running in the DAQ node's container, then killed it with this
toolkit's `emergency_stop.py` over plain SSH -- confirmed dead via `ps`/
`pgrep` on the DAQ host and via the new stack's own `pseti stat`, with
the `daqnode-server` container itself untouched -- and separately stopped
Quabo data flow over raw UDP.

## Why this exists, and why it needs no manual reconfiguration

The old software's config loader (`utils/pydantic_config_models.py`)
originally rejected any config file it didn't recognize every field of
(`extra='forbid'`) -- so it couldn't read the new stack's config files at
all (they carry extra fields like `docker_context`, `grpc_port`,
`port_forwarding.grpc_port` that the old schema never had). The three
models that gained fields (`DaqNodeValidator`, `DaqConfigValidator`,
`PortForwarding`) are patched here to `extra='ignore'` instead: unknown
fields are silently dropped rather than aborting. That's the whole fix --
this fallback reads **the exact same live config files** the new `pseti`
CLI does, via `PSETI_CONFIG` (see `_config_bridge.py`), so there is no
separate config directory to keep in sync and no reconfiguration step at
the moment you actually need this.

## One-time setup

```bash
# Python 3.9 required (the old control software's only supported interpreter)
conda create -n panoseti-legacy python=3.9 -y
conda activate panoseti-legacy
pip install -r control/legacy/requirements-legacy.txt

# Stage the DAQ-side scripts (stop_daq.py, util.py, ...) onto every
# configured DAQ node's data_dir. Re-run whenever daq_config.json's node
# list changes. Requires passwordless SSH to every DAQ node (already a
# prerequisite of the new stack's own pseti admin/pseti test hw tooling).
export PSETI_CONFIG=/path/to/active/configs   # same value `pseti env` reports
python3 control/legacy/stage_daq_nodes.py
```

## Using it

```bash
conda activate panoseti-legacy
export PSETI_CONFIG=/path/to/active/configs   # same value `pseti env` reports
python3 control/legacy/emergency_stop.py
```

This SSHes to every DAQ node and runs the staged `stop_daq.py` (SIGINT,
then SIGKILL any stragglers, by process name -- no pidfile handoff from
the new stack is assumed or required), then sends a real UDP command to
every configured Quabo to stop its data flow. Neither step touches gRPC,
HV, or power.

Options:
- `--skip-quabos` -- stop DAQ/hashpipe only, leave Quabo data flow as-is.
- `--quabo-uids PATH` -- explicit path to `quabo_uids.json`, if it isn't
  under `$PSETI_TMP` or `$PSETI_CONFIG` (the two locations auto-detected).

## What's vendored here, and what isn't

| Included | Not included |
|---|---|
| `stop.py` (`stop_recording`, `stop_data_flow`) | `start.py`, `session_start.py`, `session_stop.py` |
| `daq_scripts/{start_daq,stop_daq,status_daq}.py` | `config.py`, `power.py`, `get_uids.py`, calibration tooling |
| `driver/quabo_driver.py` | `daemons/` (Redis/InfluxDB/HK capture daemons) |
| `utils/{util,config_file,pydantic_config_models,global_validator,config_validator,collect,pff,file_xfer}.py` | `web/`, `analysis/`, anything gRPC-related |

`daq_scripts/start_daq.py` is staged onto DAQ nodes alongside `stop_daq.py`
only because the old software always deployed them together (they share
`util.py`); it is not meant to be invoked as part of this fallback.

## Known limitation

`config_file.py`'s `dir=` parameter combines with hardcoded
`'configs/<file>.json'`-style relative filenames -- it can't be pointed at
an arbitrary directory directly. `_config_bridge.py` works around this by
symlinking `PSETI_CONFIG` in as `configs` under a local scratch directory
(`control/legacy/_config_bridge/`, gitignored) rather than requiring a
manual symlink -- this happens automatically on every run.
