---
name: working-with-pff-data
description: Use when reading, iterating, parsing, or converting raw PFF data files captured by hashpipe — frame headers and binary image blocks, per-frame metadata or timestamps, the pypff library, or PFF→Zarr conversion.
---

# Working with PFF Data

## Overview

PFF (PanoSETI File Format) is written by hashpipe on DAQ nodes. Three implementations exist — know which to use. `pypff` is the high-performance analysis reader; `control/src/control/utils/pff.py` is the canonical reference/timing parser.

## Three-implementation map

| Implementation | Role | When to use |
|----------------|------|-------------|
| `util/pff.cpp` | On-disk format definition; hashpipe writer | Read to understand the wire format |
| `control/src/control/utils/pff.py` | Canonical reference Python parser + timing conversions (`wr_to_unix_precise`) | Timing math, format questions, DAQ-side tooling |
| `pypff` (import `pypff`) | High-performance mmap+NumPy+Pydantic reader | Analysis pipelines, notebooks, Zarr conversion |

**`pypff` is NOT a wrapper of `util/pff.cpp`** — it is an independent pure-Python implementation.

## Timing authority rule

For event-time reconstruction always defer to `wiki_docs/Precise-Timing.md` and `control/src/control/utils/pff.py::wr_to_unix_precise`. Do NOT trust ad-hoc timing logic in other files.

## pypff modern API (`pypff.io2`)

```python
import pypff

# Open files or a full run directory
seq = pypff.PFFSequence(["/path/to/file.pff"])
seq = pypff.PanosetiRun("/path/to/run_dir").get_product("img16")

# Iterate
for img in seq: ...                          # img is ndarray
frame, header = seq.get_frame_validated(i)   # (QuaboHeader|ModuleHeader, ndarray)

# Efficient metadata extraction
meta = seq.get_metadata_arrays(["pkt_num", "tv_sec", "pkt_nsec", "quabo_num"])
ts = seq.timestamps()                        # int64 ns; as_datetime=True for datetime64
seq.seek_time(ns)
```

Pydantic header models: `pypff.QuaboHeader`, `pypff.ModuleHeader`, `pypff.FrameConfig`.

For batch iteration, housekeeping (`hkpff`), and config (`qconfig`) see `pypff/example/pypff_io2_demo.ipynb`.

Legacy byte-offset reader: `pypff.io.datapff` (back-compat only; prefer `io2`).

## PFF file structure

Each file is a sequence of alternating blocks:
- **JSON header block**: starts with `{`, ends with `\n\n`, fixed-size (padded with spaces). Fields: `quabo_num`, `pkt_num`, `pkt_tai`, `pkt_nsec`, `tv_sec`, `tv_usec`.
- **Binary image block**: preceded by `*`, then raw pixel data.

Files roll at ~1 GB. Naming: `start_{ISO}.dp_{product}.bpp_{bytes}.dome_{N}.module_{N}.seqno_{N}.pff`.

## PFF→Zarr conversion

```python
from pypff.zarr import convert_run, PanosetiZarrRun
convert_run("/path/to/run_dir", "/output/zarr")
```

## Examples and notebooks

- `pypff/example/pypff_io2_demo.ipynb` — primary io2 API demo
- `pypff/example/pypff_example.py` — runnable script (hk, ph256, img16, PanosetiRun)
- `control/notebooks/pff_to_zarr_demo.ipynb` — PFF→Zarr workflow

## Full references

- `pypff/README.md`, `pypff/CLAUDE.md` — package overview, usage
- `wiki_docs/Data-file-format.md` — format spec (header+image blocks)
- `wiki_docs/Data-file-names.md` — directory/file naming convention
- `wiki_docs/Data-types.md` — image products (img8/img16/ph256/ph1024)
- `wiki_docs/Precise-Timing.md` + `control/src/control/utils/pff.py` — canonical timing math
- Cross-reference `working-with-quabo-driver` (packet source), `developing-control-code` (DAQ data flow)
