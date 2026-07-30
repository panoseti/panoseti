---
name: panoseti-orientation
description: Use when starting any task in the panoseti observatory codebase and unsure where code, tests, or reference docs live, or which subsystem (control, grpc, daq, analysis) or which CLAUDE.md / wiki page applies.
---

# PanoSETI Codebase Orientation

## Overview

PanoSETI is a multi-node observatory control and data-acquisition system. The repo is divided into Python-heavy subsystems with their own CLAUDE.md files and a shared `wiki_docs/` GitHub wiki mirror.

## When NOT to use

If you already know which subsystem you're in, skip this skill and load the relevant topic skill directly from the table below.

## Repo map

| Dir | Contents |
|-----|---------|
| `control/` | Observatory control plane: `pseti` CLI, run lifecycle, config, daemons, hardware drivers |
| `grpc/` | Git submodule → `panoseti_grpc`: daq_data, daq_control, telemetry gRPC services |
| `pypff/` | High-performance PFF file reader (mmap + NumPy); `pypff` package |
| `daq/` | Git submodule → Hashpipe DAQ pipeline (C/C++) |
| `analysis/` | Data analysis framework |
| `util/` | Shared C++ utilities: `pff.cpp` (PFF writer/reader), image processing |
| `wiki_docs/` | GitHub wiki mirror — markdown reference docs (read-only authoritative) |
| `alloy/` | Grafana Alloy config + Docker Compose for Loki log shipping |

## CLAUDE.md files and their scope

| File | Owns |
|------|------|
| `CLAUDE.md` (root) | Hardware topology, config system, observing run flow, PFF format, timing, gRPC service overview |
| `control/CLAUDE.md` | Transaction/lock model, Pydantic conventions, test tiers, telemetry logger |
| `grpc/CLAUDE.md` | gRPC architecture, service internals, testing infra, key gotchas |
| `grpc/GEMINI.md` | Engineering standards, concurrency rules, coding style |
| `pypff/CLAUDE.md` | pypff package internals |

## wiki_docs/ domains

`wiki_docs/` is the GitHub wiki mirror (~50 `.md` files). Search it by keyword; the files are named after topics. Key domain groupings:

| Domain | Topics covered |
|--------|---------------|
| Hashpipe / DAQ | Hashpipe overview, data-recorder threads, DAQ-system-overview, building hashpipe, test data |
| Config / setup | Configuration-files, Sessions-and-configuration, network-configuration, node setup, storage |
| Data format | Data-file-format, Data-file-names, Data-types, Metadata, Pixel-indexing |
| Hardware / Quabo | Quabo-packet-interface, Quabo-device-driver, Nodes-and-modules, TFTP, power control |
| Observing / timing | Observing-runs, Precise-Timing, Interleaving, White-Rabbit-* |
| Analysis | Analysis-framework, Pulse-finding-* |
| Control system | Control-system-implementation, Web-summary |

## Load one of these skills next

| Task | Skill |
|------|-------|
| Running or scripting `pseti` commands / observing lifecycle | `using-pseti-cli` |
| Running, writing, or debugging tests / CI | `testing-panoseti` |
| Modifying gRPC services or adding a new one | `developing-grpc-services` |
| Modifying control/ code (transactions, config, daemons) | `developing-control-code` |
| Sending UDP commands to Quabo boards / MAROC / HV | `working-with-quabo-driver` |
| Reading, parsing, or converting PFF data files | `working-with-pff-data` |
