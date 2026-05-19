---
name: working-with-quabo-driver
description: Use when sending commands to or reading packets from Quabo detector boards — the UDP command/housekeeping interface, the QUABO driver class, DAQ_PARAMS / acquisition-mode bits, the MAROC serial command, HV/baseline calibration, or firmware/TFTP reboot.
---

# Working with the Quabo Driver

## Overview

The Python UDP device driver for Quabo detector boards lives in `control/src/control/driver/`. The main class `QUABO` sends binary command packets over UDP and reads housekeeping replies.

## Driver directory

| File | Purpose |
|------|---------|
| `quabo_driver.py` | `QUABO` class, `DAQ_PARAMS`, `get_daq_params()`, packet helpers |
| `quabo_tftp.py` | `tftpw` class — TFTP firmware flash + FPGA reboot |
| `quabo_config.txt` | Default MAROC register config (`name=value`, parsed by `parse_quabo_config_file`) |

## UDP port map

| Port | Direction | Purpose |
|------|-----------|---------|
| `UDP_CMD_PORT = 60000` | Head node → Quabo (and echo reply back) | Commands |
| `UDP_HK_PORT = 60002` | Quabo → Head node | Housekeeping packets (~3 s interval) |
| Science port 60001 | Quabo → DAQ node | Science data (not received by head node) |

Science packet destination is set per-quabo via `QUABO.data_packet_destination(daq_ip)` (cmd `0x0a`).

## QUABO class API surface

`QUABO(ip_addr, port=UDP_CMD_PORT, config_file_path='quabo_config.txt')`

Key method groups (full signatures in `quabo_driver.py` and `wiki_docs/Quabo-device-driver.md`):
- **Acquisition**: `send_daq_params(params)`, `send_maroc_params()`, `send_maroc_params_file(path)`
- **HV / calibration**: `hv_on/off()`, `hv_set(vals)`, `hv_set_chan(chan, val)`, `calibrate_ph_baseline()` → 256 coefficients
- **Packet routing**: `data_packet_destination(daq_ip)`, `hk_packet_destination(ip)`
- **Housekeeping**: `read_hk_packet()` — blocks up to 10 s, filters by source IP
- **Hardware**: `reset()`, `reboot()`, `shutter_new(open=True/False)`, `focus(steps)`, `fan(speed)`, `close()`

## DAQ_PARAMS and acquisition mode bits

`DAQ_PARAMS(do_image, image_us, image_8bit, do_ph, bl_subtract, do_any_trigger, do_group_ph_frames)` — or build from Pydantic `DataConfig` via `get_daq_params(data_config)`.

Mode byte (cmd `0x03`): `ACQ_PULSE_HEIGHT=0x1`, `ACQ_IMAGE=0x2`, `ACQ_IMAGE_8BIT=0x4`, `ACQ_NO_BASELINE_SUBTRACT=0x10`.

## MAROC serial command

`make_maroc_cmd` builds a 829-bit (`SERIAL_COMMAND_LENGTH`) shift-register packet from `quabo_config.txt` tag names. For protocol internals see `wiki_docs/Quabo-packet-interface.md`. Always call `flush_rx_buf()` before sending; echo variants use MS-bit command codes (`0x80`).

## Key callers in control/

`start.py` (DAQ params + data destination), `config.py` (HV, MAROC, calibration, masks), `utils/util.py` (probe/quiet quabos), `tools/{shutter,focus_step,hv_updater,interleave}.py`.

## Module/quabo IP math

Defer to root `CLAUDE.md` → Key Conventions (module ID from IP, quabo index within module).

## Full references

- `wiki_docs/Quabo-packet-interface.md` — authoritative wire protocol spec (packet layouts, command bytes)
- `wiki_docs/Quabo-device-driver.md` — QUABO class usage guide
- `wiki_docs/Nodes-and-modules.md` — module/quabo numbering
- `control/src/ci/software_only_v2/tier1_unit/test_quabo_driver*.py` — behavioral test reference
- Cross-reference `working-with-pff-data` for packets after they reach disk
