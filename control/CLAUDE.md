# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file supplements the root-level `../CLAUDE.md`, which covers the full repo architecture, hardware topology, config system, and observing run lifecycle. Read that first for context.

---

## Corrections to root CLAUDE.md

The root CLAUDE.md has some stale entries for the `control/` package:

- **Python version**: `requires-python = ">=3.14"` (not 3.9)
- **CI runner**: `python ci/qa.py <cmd>`
- **Integration test count**: 69 passing
- **Unit test count**: 475 passing
- **Chaos/scenario test count**: 113 tests (66 active, 47 stubs) in `ci/integration/scenarios/`

---

## Verification & Quality Standards

### Linting and Type Safety
The project enforces strict linting via Ruff and type checking via MyPy. All new code must pass `python ci/qa.py lint`.

- **Pydantic Model Authority**: Instantiated models from `utils/pydantic_config_models.py` must be passed across call boundaries. Polymorphic functions must validate dictionaries into models at the entry point.
- **Attribute Access**: Always prefer model attribute access (`config.daq_nodes`) over dictionary indexing (`config['daq_nodes']`).
- **MyPy Strictness**: Avoid `type: ignore` whenever possible. If required, use it on a specific line with a comment explaining why. Ensure `unused-ignore` rules pass.

### Documentation (Google Style Docstrings)
All functions and methods must have high-quality docstrings. Preserving legacy comments (prefixed with `#`) by transforming them into formal docstrings is mandatory.

```python
def example_function(arg1: int, arg2: str) -> bool:
    """A concise summary of the function's purpose.

    Detailed description of the intent and spirit of the operation, 
    preserving any relevant implementation notes from legacy comments.

    Args:
        arg1: Description of the first argument.
        arg2: Description of the second argument.

    Returns:
        True if the operation succeeded, False otherwise.
    """
```

### Testing and Debugging
- **Unit Tests**: Add new cases to `ci/unit/` for every utility function. No hardware or network access is allowed in unit tests.
- **Integration Tests**: Verify end-to-end flows in `ci/integration/`. Use the `-k` flag to isolate failures (e.g. `python ci/qa.py integration -k TestConfigValidation`).
- **Transactional Debugging**: If a run fails, inspect `tmp/run_state.toml` and check `_aborted/` for failure context dumps.
- **Advisory Locking**: When debugging control scripts standalone, ensure they acquire the advisory lock on `tmp/panoseti_control.lock` or use `RunStateManager`.
- **Async Safety**: Use `asyncio.to_thread` for blocking file system or socket I/O within async functions to prevent event loop starvation.

Read control/DEBUGGING.md for more advanced debugging advice for the control/ directory.


### Install (native, without Docker)

```bash
cd control
uv sync --all-extras         # preferred — uses uv.lock
# or: pip install -e ".[dev]"
```

### Run tests

```bash
# Docker-based (preferred — matches CI exactly)
python ci/qa.py up           # start persistent background containers once
python ci/qa.py unit         # ~3s, parallel
python ci/qa.py integration  # ~75s, E2E with real hashpipe
python ci/qa.py lint         # ruff + mypy concurrently
python ci/qa.py down         # tear down

# Targeted test runs (pass any pytest args after the command)
python ci/qa.py unit -k test_pff
python ci/qa.py integration -k TestDaqLifecycle

# Native (no Docker, unit tests only)
uv run pytest ci/unit/
uv run pytest ci/unit/ -k test_config_file -v --tb=short

# Real hashpipe + tcpreplay tests (requires running containers)
python ci/qa.py integration -k "real_data"

# Chaos/TDD-forcing scenario tests (expected to fail red on master)
python ci/qa.py chaos                          # all 113 chaos tests
python ci/qa.py chaos -k "SC010 or SC002"      # specific exemplars
```

The `chaos` command runs `ci/integration/scenarios/` only, which is **excluded** from the normal `integration` suite. This means `python ci/qa.py integration` stays green while `python ci/qa.py chaos` is expected to have failures on unimplemented features.

### Lint and type-check

```bash
# Via qa.py (recommended)
python ci/qa.py lint

# Direct
ruff check utils/ driver/
mypy utils/config_file.py --ignore-missing-imports
```

---

## Ruff and MyPy Configuration

Defined in `pyproject.toml`:

- **Ruff rules**: `E`, `F`, `UP`, `B`, `I`, `ASYNC`, `RUF`, `SIM` — 100-char line length; `E501` and `B008` ignored
- **MyPy**: strict-ish (`disallow_untyped_defs`, `check_untyped_defs`, `warn_unreachable`); pydantic plugin enabled; `ci.*` module overrides relax type requirements for test fixtures

---

## CI Architecture Notes

- **Persistent containers**: `python ci/qa.py up` starts containers with `sleep infinity`. They are reused across runs — no boot overhead per test invocation.
- **Live mount**: `control/` is volume-mounted into `/app` in containers. Source edits are immediately visible to the next test run without a rebuild.
- **Dependencies isolated**: Python env lives in `/opt/venv`, separate from the live mount. Only rebuild (`python ci/qa.py build`) when `pyproject.toml` or `uv.lock` changes.
- **BINDHOST=lo**: Hashpipe always binds to loopback in CI. `tcpreplay` injects PCAP packets into `lo` to bypass MAC filtering and MTU limits on Docker virtual NICs.

Integration network topology: `headnode_net` (10.0.1.0/24) with pytest runner, gateway, Redis, Loki, headnode Telemetry service; `daqnode_net` (192.168.0.0/24) with two DAQ nodes. See `ci/README.md` for the full diagram.

---

## Session Start Stages

`session_start.py` supports resuming from any stage — useful when debugging mid-session:

```
poweron → get_uids → reboot → hk_dest → redis → maroc → ph_baseline
```

Pass `--stage <name>` to skip earlier steps.

For daytime debugging without detector HV:

```bash
python session_start.py --no_hv
```

For automated timed runs (e.g. sleep after starting):

```bash
python start.py --nsecs 7200 --end_session   # stops recording + ends session after 2 h
```

---

## Interleave Mode Lifecycle

Interleaving is **an overlay on top of a normal run**, not an automatic behavior of `start.py`. Order:

```bash
python config.py --validate        # or --validate network for reachability checks
python start.py                    # boots the default image + pulse_height modes
python config.py --start-interleave   # launches tools/interleave.py as a background daemon
# ...
python config.py --stop-interleave    # graceful stop (also triggered automatically by stop.py)
python stop.py
```

Key points:
- The interleaver writes `tmp/interleave.pid` and reads `data_config.json` `interleave.states[]` to cycle modes.
- Each state references `movie_mode_config` and/or `pulse_height_mode_config` by key (top-level `image_*` or `pulse_height_*` entries); set to `null` to disable that data product during that state.
- Constraints enforced by Pydantic: a state cannot set *both* to `null`; movie mode cannot coexist with `two_pixel_trigger`/`three_pixel_trigger > 0` in the same state.
- Quabos need ~100 ms to transition between states → the first ~100 ms of movie frames after a switch have missing quabos ("partial images"). Tests asserting frame completeness must allow for this.
- `stop.py::stop_interleave()` calls `os.kill(pid, SIGTERM)` on the PID in `tmp/interleave.pid`, polls for PID file removal up to 10 × 0.5 s, then gives up silently — there's no hard-kill fallback, and no verification that the PID actually belongs to our interleaver (stale PID → wrong process could be signaled).

---

## Storage Topology

### DAQ node storage (`Storage-on-DAQ-nodes.md`)

DAQ nodes spread module data across multiple disks **without RAID**. In `~panoseti/data/`, `module_N/` may be a plain directory on the main disk *or* a symbolic link to `/mnt/diska/data/module_N` (or similar) on a secondary disk. No JSON config tracks this; the control scripts only create `module_N/` on the main volume if the symlink is absent. ENOSPC, `os.walk`, and `rsync` behavior differ between these two layouts — anything that asserts against "the data dir" must follow the symlink.

### Head node storage (`Storage-on-the-head-node.md`)

The head node may have multiple **PanoSETI volumes**, each a directory containing `data/` and `analysis/` subdirs (e.g. `/home/panosetigraph/panoseti_data/`, `/mnt/data10/`, `/mnt/data11/`). `daq_config.json::head_node_data_dir` selects which volume the current run writes to. Operators switch to a different volume when the current one fills up; there is no automatic fallback.

All volumes are symlinked under `/home/panosetigraph/web/` for web dashboard access.

---

## PFF File Format (detailed)

The on-disk layout matters for test assertions and mmap/stride-based readers (see `../sw-multi-pix-pulse-height/panoseti_interface.py`):

| Data product | JSON header | Binary payload | Frame size | Image shape / dtype |
|---|---|---|---|---|
| `ph256`   | 124 bytes  | 512 bytes   | 124 + 1 + 512  = 637  | (16, 16) int16 |
| `ph1024`  | 492 bytes  | 2048 bytes  | 492 + 1 + 2048 = 2541 | (32, 32) int16 |
| `img8` (mov-8)  | 492 bytes | 1024 bytes | 492 + 1 + 1024 = 1517 | (32, 32) uint8 |
| `img16` (mov-16) | 492 bytes | 2048 bytes | 492 + 1 + 2048 = 2541 | (32, 32) uint16 |

**Layout per frame**: `{json...}\n\n` (padded to fixed size) + `*` byte + raw pixel bytes.

**Fixed-frame invariant**: after the first frame in a file, **all JSON header blocks have identical length** (padded with spaces). This enables O(1) seek, mmap+stride reads, and binary search on timestamps. Any test that writes PFF data must preserve this padding; any test that reads must compute frame size from the first frame only.

`ph256` header is per-quabo; `ph1024` and `mov-*` headers wrap four quabo sub-headers (`quabo_0..quabo_3`, each with its own `pkt_num`, `pkt_tai`, `pkt_nsec`, `tv_sec`, `tv_usec`). Use `quabo_0.tv_sec` (or the first non-zero quabo) for the module-level timestamp.

---

## Precise Timing

**Authoritative source**: `../../panoseti-docs/Precise-Timing.md` and `control/utils/pff.py`. These were written and vetted by the timing experts and are the reference for every downstream consumer. Do **not** substitute `sw-multi-pix-pulse-height/panoseti_interface.py` — its 50 ms threshold is unvetted.

Each science packet carries two timestamps:
- `pkt_nsec` (= `NANOSEC` in raw packets) — WR/GNSS nanosecond-within-second, ns-accurate, event-triggered at the quabo.
- `tv_sec` + `tv_usec` — UNIX time from the DAQ node (NTP-synced via the head node), accurate to ~µs–ms.

The DAQ node must be NTP-locked (`chronyc sources` must show a valid sync) before precise times are meaningful.

Algorithm (per `Precise-Timing.md`), using a **25 ms** tolerance between `tv_usec * 10³` and `pkt_nsec`:

| Relationship | Precise time |
|---|---|
| `|tv_usec·10³ − pkt_nsec| ≤ 25 ms` | `tv_sec + pkt_nsec / 10⁹` |
| `tv_usec·10³ ≫ pkt_nsec` (NTP is 1 s behind GPS) | `tv_sec + 1 + pkt_nsec / 10⁹` |
| `pkt_nsec ≫ tv_usec·10³` (NTP is 1 s ahead of GPS) | `tv_sec − 1 + pkt_nsec / 10⁹` |

Timing tests assert against this exact rule. Reference implementation: `control/utils/pff.py::img_header_time` (around line 238).

---

## Config File Conventions

The active config set is a **symlink** to a site-specific file. The pattern for all config files:

```bash
ln -s daq_config_lick_2nodes.json daq_config.json
```

Keep named variants in the repo (e.g. `daq_config_palomar_2node.json`) and commit them. Never edit the symlink target for a site you share with others.

**Non-obvious fields by file:**

`obs_config.json`:
- `quabo_version` can be a mixed array per module: `["qfp", "qfp", "bga", "qfp"]`
- `timing_mode` defaults to `"wr"` if absent
- `wps` key in a module entry names which web power switch controls it (default `"wps"`)

`daq_config.json`:
- `head_node_container: true` skips the IP reachability check for the head node (used in Docker CI)
- `bindhost` per DAQ node defaults to `"0.0.0.0"`; some hardware requires `"eno1"` or `"eth0"`
- `module_ids` supports ranges: `"0-127"`

`data_config.json`:
- `run_type` must be `"science"`, `"engineering"`, or `"calibration"` (max 14 chars, no `.` `_` or spaces)
- `nsum`: sum N image frames per output frame in image mode
- `pulse_height.any_trigger.group_ph_frames: 1` makes Hashpipe group all 4 quabo PH packets per module into one frame
- Interleave states use `null` (not `false`) to disable a mode for that state

`daemons.json` (daemon_config.json):
- Short keys map to `capture_<key>.py`: `"hk"` → `capture_hk.py`, `"gps"` → `capture_gps.py`, etc.

**Generated/calibration files** (not hand-edited):
- `quabo_uids.json` — generated by `get_uids.py`; empty UID string means that quabo is absent or broken
- `quabo_calibration/quabo_calib_<UID>.json` — one file per quabo; contains `pixel_gain_delta[256]` and per-quadrant `a`, `b` (for DAC1 = `int(a*(gain*PE_level) + b)`) and `n`, `m` (for ADC = `m*gain*PE_level + n`)

---

## config.py CLI Flags

`config.py` is the multi-purpose hardware control script called by `session_start.py` but also useful standalone:

| Flag | Action |
|------|--------|
| `--show` | List domes/modules/quabos from config |
| `--ping` | Ping all quabos |
| `--reboot` | Reboot quabos |
| `--loads` | Load silver firmware |
| `--hv_on` / `--hv_off` | Enable/disable detector high voltage |
| `--maroc_config` | Write MAROC DAC/gain registers |
| `--mask_config` | Set trigger and GOE masks |
| `--calibrate_ph` | Run PH baseline calibration |
| `--hk_dest` | Direct HK packets to the head node |
| `--redis_daemons` | Start metadata daemons (GPS, WR, HK → Redis) |
| `--stop_redis_daemons` | Stop the above |
| `--init_daq_nodes` | Copy current software to DAQ nodes |
| `--disk_space` | Show remaining recording hours on DAQ/head nodes |

---

## Quabo Packet Encoding

Key facts for working on `driver/quabo_driver.py` or `utils/pff.py`:

**Command port**: UDP port 60000. MS bit of command byte set to `0x80` requests an echo response.

**`acq_mode` byte** (Set Acquisition command `0x03`, science packet header byte 0):

| Value | Mode |
|-------|------|
| `0x00` | Disabled |
| `0x01` | PH mode (with BL subtract) |
| `0x11` | PH mode (no BL subtract) |
| `0x02` | 16-bit Image mode |
| `0x03` | Simultaneous 16-bit IM + PH |
| `0x06` | 8-bit Image mode |
| `0x07` | Simultaneous 8-bit IM + PH |

**Science packet sizes**: 528 bytes (16-bit or PH mode), 272 bytes (8-bit mode). Header is 16 bytes; pixel data starts at offset 16.

**NANOSEC field**: encoded in 3.125 ns ticks (320 MHz clock), not nanoseconds. Convert: `nanosec_actual = NANOSEC * 3.125`.

**MAROC setup**: 829-bit shift register per chip, sent as 107 bytes per ASIC in the Set ASICs command (`0x01`, 492-byte packet). Echo response returns readback values — the *first* readback after power-on will not match what was written; subsequent identical loads should match.

**HV encoding**: steps of −1.14 mV/LSB; −75 V max. `HVIMON` current: `I_uA = (65535 − HVIMON) × 0.0381`, minus `HVmon_volts / 0.499` offset.

**GOE mask** (Set GOE masks command `0x0e`): `0x3` = any single pixel trigger; `0x1` = 2+ pixels; `0x2` = 3+ pixels.

**HK packet**: 64 bytes on UDP port 60002, emitted every ~3 s. `bootbyte = 0xaa` on first packet after CPU boot, then 0. `BOARDLOC = module_id * 4 + quadrant_index`.
