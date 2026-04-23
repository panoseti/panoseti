# PANOSETI Integration Test Design Document

**Status:** Draft for review
**Author:** Claude (Principal QA Architect role, Opus 4.7)
**Intended reader:** Mid-level engineer implementing the pytest suite
**Scope:** `control/` + `panoseti_grpc/` coordinated end-to-end testing

---

## Context

PANOSETI is transitioning its observatory control plane from brittle SSH-driven RPCs to a gRPC-coordinated distributed system. The current critical path — `start.py`, `stop.py`, and the data-collection handoff — has **no transactional guarantees**: any mid-flight failure (network hiccup, hashpipe crash, disk full, operator ctrl-C) leaves the telescope array in a partially-configured state. Quabos may be streaming data into a void, hashpipe processes may be orphaned, or the head node may believe a run is active while DAQ nodes have cleaned up.

This test suite has two goals:

1. **Force a TDD rewrite** of `start.py` / `stop.py` / `config.py` critical paths so that every multi-step operation is either atomic-with-rollback or safely idempotent. Tests must be designed so they **definitively fail on the current codebase** and only pass once transactional logic is added.
2. **Establish a regression harness** that catches silent breakage of the observatory's operational envelope, both in the cloud digital twin (fast feedback) and on the mountain (reality check).

The existing `ci/docker-compose.integration.yml` stack (headnode + 2 DAQ nodes + socat gateway + Redis + Loki) is the foundation. This document expands it into a full chaos-engineering and HITL framework.

---

## Non-negotiable principles

| Principle | Rationale |
|---|---|
| **No return-code gating on quabo commands.** | `quabo_driver.QUABO` sends UDP without ACKs — a failure is the *absence of effect*, not a false return. Tests must simulate via exceptions, timeouts, PCAP replay divergence, or post-hoc state assertions. |
| **Every failure injection reveals a current bug.** | If a chaos test passes on `master` today, either the injection is weak or the test is redundant. Each case must be paired with an expected-vs-current behavior annotation (see Appendix A). |
| **State assertions outweigh log assertions.** | Logs lie; Redis keys, PID files, `current_run`, PFF headers, and the Hashpipe shared-memory segment do not. |
| **Every chaos fixture is reversible.** | Tests must never leave the docker stack in a state that poisons the next test. All fixtures use `try/finally` with explicit teardown of `tc qdisc`, iptables rules, monkey-patches, and killed processes. |
| **The software-only pillar is the gate; HITL is the proof.** | CI blocks merges on Pillar 1. HITL runs nightly on the mountain and blocks deploy tags. Pillar 3 (scaling) runs on-demand before capacity changes. |
| **Timing logic follows `Precise-Timing.md` / `control/utils/pff.py`.** | These were authored by the timing experts and are authoritative. Do **not** trust `sw-multi-pix-pulse-height/panoseti_interface.py` — its 50 ms threshold and 500 ms wrap-detect are unvetted. Canonical threshold is **25 ms** between `tv_usec·10³` and `pkt_nsec`; `tv_sec` is adjusted ±1 s on wrap. |
| **Interleave is a separate lifecycle stage, not part of `start.py`.** | `python config.py --start-interleave` is invoked **after** `start.py` succeeds, launching `tools/interleave.py` as a background daemon (PID file: `tmp/interleave.pid`). `stop.py` calls `stop_interleave()` automatically; `config.py --stop-interleave` is the manual path. Tests must not assume interleave is triggered by `start.py`. |
| **PFF frames are fixed-size after frame 0.** | Every JSON header block in a given file is padded to the length of the first header. Tests that write or mutate PFF must preserve padding; tests that read compute frame size from frame 0. |
| **DAQ node `module_N/` may be a symlink.** | For multi-disk write striping, `~panoseti/data/module_N/` can point to `/mnt/diska/data/module_N`. No JSON tracks this. Disk-fill tests, `os.walk`, and `rsync` assertions must resolve symlinks. |
| **Head node has multiple PanoSETI volumes.** | Each volume is a directory (e.g. `/home/panosetigraph/panoseti_data/`, `/mnt/data10/`) with its own `data/` and `analysis/`. `daq_config.json::head_node_data_dir` picks one per run. Operators switch volumes manually when one fills; no automatic fallback. |

---

## Pillar 1 — Software-Only Critical Path (Chaos Suite)

### 1.1 New infrastructure required

#### 1.1.1 `mock_quabo` service

The current `docker-compose.integration.yml` has no quabo simulator — tests rely on `tcpreplay` PCAP injection for science packets but offer nothing for the **command-plane** UDP (ports 60000-60003, plus 60002 HK echo). Without this, `start.py`'s `start_data_flow()` calls to `quabo_driver.QUABO.send_daq_params()` fire into a void, and there is no way to assert "command was received and acknowledged" or "command timed out."

Add a new container `mock-quabo` to the stack.

**Scope (per Appendix B-6):** `mock_quabo` models **only the packet interface** documented in `Quabo-packet-interface.md`. It does not simulate firmware bugs, hardware errata, or correctable-but-flaky behavior. Software-only tests assume ideal hardware so failures are attributable to the control plane. Chaos injection lives in network / process / disk / timing primitives, not inside the simulator.

- **Image:** `ci/mock_quabo/Dockerfile` — thin Python 3.14 service built on the `base` stage.
- **Module:** `ci/mock_quabo/server.py` — asyncio UDP server that:
  - Binds UDP `:60000` on a configurable IP list (one per simulated quabo).
  - Parses the command byte per `Quabo-packet-interface.md` (`0x81`/`0x01` SetASICs → echo 492 bytes; `0x82` Set HVs; `0x83` Set Acquisition; `0x84` Reset; `0x86` Channel Mask; `0x07` Calibrate PH Baseline → 516-byte response; `0x8f` Software 1PPS; `0x20` HK interval).
  - Emits HK packets (64 bytes, offset `0x20`) on UDP `:60002` every 3 s to the `hk_dest` IP it has been told, including plausible `BOARDLOC`, `HVMONx`, `TEMP1`, `UID`, `FWVER`.
  - Responds to `SetASICs` deterministically: readback always equals the last-written value (ideal-hardware contract). No `CHAOS_ASIC_MISMATCH` mode.
  - Exposes a **control socket** (UDS at `/tmp/mock_quabo.sock`) for tests to drive topology-level state — *not* firmware-level misbehavior. Valid commands: `set_hk_dest <ip>`, `report_state`, `reset`, `emit_science_packet <header_json>` (see below).
- **Science-data generation:** the default pattern for streaming science UDP is **`tcpreplay` injecting canned PCAPs into loopback**, matching the current `docker-compose.integration.yml`. For a handful of targeted tests (e.g. timing-boundary SC-054/SC-055, fixed-frame invariant SC-049b) where maintaining PCAPs is overkill, `mock_quabo` may emit a small number of science UDP datagrams on command via `emit_science_packet`. Any test using the emitter must construct the exact 528-/272-byte packet body itself, so headers and `NANOSEC`/`BOARDLOC` fields are test-authored and inspectable.
- **Network:** joins `daqnode_net` with an alias IP per quabo. For 1 dome × 2 modules × 4 quabos = 8 simulated IPs on `192.168.3.248-251` and `192.168.3.252-255`.
- **Volumes:** mounts `ci/integration/pcaps/` read-only so `tcpreplay` sidecars can replay canned UBX/HK PCAPs when asked.

#### 1.1.2 Chaos control plane

Add `ci/integration/chaos/` with helpers:

| File | Purpose |
|---|---|
| `netem.py` | wraps `tc qdisc add dev <iface> root netem` to add loss/latency/duplicate. Requires `cap_add: NET_ADMIN` (already present on daqnode containers). |
| `iptables.py` | blackhole specific dst IPs or ports (`iptables -A OUTPUT -d 192.168.3.250 -j DROP`). |
| `grpc_proxy.py` | pytest fixture that spawns a `grpcwebproxy`-style interceptor between `int-tester` and the target DAQ node. Modes: `timeout`, `unavailable`, `success_then_fail`, `slow_response`, `partial_response`, `reset_stream`. |
| `process_chaos.py` | `docker exec` helpers to `SIGKILL`, `SIGSTOP`, `SIGTSTP` a named process by `pidof` inside a container (hashpipe, panoseti-server, interleave). |
| `disk_chaos.py` | `dd if=/dev/zero of=/data/.fill bs=1M count=N` to force ENOSPC on a specific volume. Teardown unlinks the file. |
| `clock_chaos.py` | `date -s` inside a container (needs `SYS_TIME` cap) or monkey-patches `time.time` in-process to skew WR/GNSS vs NTP by N ms. |

All chaos helpers are **context managers** that record state on enter and undo on exit even if the test body raises.

#### 1.1.3 State-inspection helpers

Tests assert against real state, not log scraping:

```python
# ci/integration/state_probe.py
class StateProbe:
    def current_run(self) -> str | None: ...          # reads /data/head/current_run
    def hashpipe_pid(self, node: str) -> int | None:  # grpc StatusDaq
    def hashpipe_process_alive(self, node: str) -> bool:  # docker exec pgrep hashpipe
    def quabo_state(self, quabo_ip: str) -> QuaboState:   # mock-quabo control UDS
    def pff_files(self, module_id: int) -> list[Path]:    # glob /data/module_N/<run>/*.pff
    def redis_keys(self, prefix: str) -> list[str]: ...
    def loki_logs(self, since: datetime, selector: str) -> list[dict]: ...
    def interleave_pid_file_exists(self) -> bool: ...
```

### 1.2 Fault-injection primitive catalog

Every Pillar 1 test is composed from these primitives. The catalog keeps the matrix tractable.

| Layer | Primitive | Implementation |
|---|---|---|
| **Quabo UDP** | silent quabo | `mock_quabo` control UDS → `silence` |
| | slow quabo | `netem delay 500ms` on daqnode egress |
| | packet drop 5% | `netem loss 5%` |
| | bad SetASICs echo | `mock_quabo` → `return_bad_echo` |
| | reboot mid-run | `mock_quabo` → `reboot_self` (HK packet sets `bootbyte=0xaa`) |
| **Hashpipe process** | crash post-start | `process_chaos.kill_after("daqnode", "hashpipe", delay_s=2)` |
| | freeze | `SIGSTOP hashpipe` |
| | orphan (stale PID) | `SIGKILL` + server still returns pid>0 from last StartDaq |
| | slow exit | `SIGTERM` ignored for 30 s by wrapper script |
| **DAQ Control gRPC** | timeout | `grpc_proxy.mode = "slow_response"` at 120 s |
| | UNAVAILABLE | `docker stop daqnode` between `StartDaq` and `StatusDaq` |
| | partial-N failure | only one of N proxies injects `UNAVAILABLE` |
| | reset stream | `grpc_proxy.reset_stream()` during `CleanupData` |
| **Telemetry** | Loki down | `docker stop loki` for 30 s mid-run |
| | Redis full | `redis-cli CONFIG SET maxmemory 10mb` then flood logs |
| | RedisBatcher stall | monkey-patch `RedisBatcher.flush_interval = 3600` |
| | storeLoki crash | `process_chaos.kill("headnode", "storeLoki.py")` |
| **Filesystem** | head node data dir ENOSPC | `disk_chaos.fill("/data/head", 99)` |
| | daqnode data dir ENOSPC | `disk_chaos.fill("/data")` on `daqnode` volume |
| | config file vanishes mid-run | `rm /app/configs/data_config.json` |
| | run dir already exists | `mkdir /data/head/<predicted_run_name>` before `start.py` |
| **Orchestration** | concurrent start | fire two `start_run()` via `asyncio.gather` |
| | stop with no run | empty `current_run`, call `stop_run()` |
| | stop while starting | kick `stop_run()` while `StartDaq` still in flight |
| | interleave zombie | `SIGKILL tools/interleave.py` but leave PID file |

### 1.3 Edge-case matrix (exhaustive)

Tests are numbered `SC-###` (Software Chaos). Each case name is also its pytest id.

#### gRPC failure isolation (SC-001 → SC-020)

| # | Scenario | TDD-forcing? | Current bug |
|---|---|---|---|
| SC-001 | `StartDaq` times out after 30 s on node 0 of 1 | yes | `start.py` has no timeout on `DaqControlClient.StartDaq`; hangs forever. |
| SC-002 | `StartDaq` UNAVAILABLE on node 1 of 2, node 0 succeeded | yes | `start_recording` raises, but node-0 hashpipe is running, quabos already streaming — no rollback. |
| SC-003 | `StartDaq` returns `success=False` (hashpipe binary missing) | yes | raises Exception, but HK recorder and HV updater already started. |
| SC-004 | `StartDaq` transient UNAVAILABLE, succeeds on retry | yes (after impl) | no retry layer exists. |
| SC-005 | `StartDaq` succeeds but hashpipe exits within 1 s | yes | no post-start liveness check; `StatusDaq` never polled. |
| SC-006 | `StopDaq` times out (wrapper won't propagate SIGINT) | yes | `stop_recording` raises on first failure, subsequent nodes never told to stop. |
| SC-007 | `StopDaq` on already-stopped service | no | should return `success=True` per design (test documents contract). |
| SC-008 | `StopDaq` on never-started service | no | contract: success, no-op. |
| SC-009 | `CleanupData` called while `hashpipe_pid > 0` | no | should error with `FAILED_PRECONDITION`; test locks contract. |
| SC-010 | `CleanupData` on orphaned PID (hashpipe SIGKILLed) | **yes** | **stuck forever**: server returns `pid>0` even after liveness lost. Requires server fix: `pid_liveness_check`. |
| SC-011 | `CleanupData` partial: node-0 succeeds, node-1 fails | yes | `stop.py._cleanup_daq_grpc` logs error, continues — but `run_complete_filename` is still written, masking loss. |
| SC-012 | `CleanupData` with full disk on head node (rsync target) | yes | `collect.collect_data` error path leaves `collect_complete_filename` unwritten, blocks cleanup correctly — but does not retry. |
| SC-013 | `StatusDaq` during `StartDaq` in-flight | no | documents RPC ordering guarantee. |
| SC-014 | gRPC channel gets `RST_STREAM` mid-`StartDaq` | yes | unclear handling; test pins behavior. |
| SC-015 | Daqnode reboots during recording | yes | head node keeps `current_run`; next `start.py` refuses because "run in progress." |
| SC-016 | `DaqControlClient` created with wrong port | yes | silent timeout, no clear error to operator. |
| SC-017 | Unified server `daq_control` toggle off on daqnode | yes | `panoseti-server --profile daq_node` without `daq_control=true` → UNIMPLEMENTED. |
| SC-018 | Concurrent `StartDaq` to same daqnode | no | server must enforce single-hashpipe-per-node; test documents it. |
| SC-019 | `CleanupData` while second `StartDaq` races with `StopDaq` | yes | no server-side mutex between cleanup and start. |
| SC-020 | gRPC deadline exceeded on `StopDaq`, SIGKILL fallback | yes | current code has no SIGKILL escalation when SIGINT is ignored. |

#### Transactional state corruption (SC-021 → SC-040)

| # | Scenario | TDD-forcing |
|---|---|---|
| SC-021 | Kill `start.py` after `make_run_dirs` before `start_data_flow` | yes — run dirs exist on head & DAQ, `current_run` unset, next start.py creates colliding run name if clock identical |
| SC-022 | Kill `start.py` after `start_data_flow` before `start_recording` | yes — quabos streaming science packets to DAQ node, but no hashpipe listening → kernel drops, data loss, not surfaced |
| SC-023 | Kill `start.py` after `start_recording` before `write_run_name` | yes — hashpipe running, head node unaware (no `current_run`); next start.py double-starts |
| SC-024 | Two `start.py` invocations in parallel | yes — no lockfile; both pass `read_run_name()` as None; both attempt `make_run_dirs` (second gets `FileExistsError`) |
| SC-025 | `start.py` with run already in progress | no — current check catches this |
| SC-026 | `stop.py` with no run in progress | no — current behavior: prints "No run in progress", exits. Pin it. |
| SC-027 | `stop.py --run X` when `current_run` says Y | yes — currently processes X, leaves Y orphaned |
| SC-028 | `stop.py` ctrl-C between `stop_recording` and `stop_data_flow` | yes — quabos still streaming to now-dead DAQ; fills kernel buffers |
| SC-029 | `stop.py` ctrl-C between `collect_data` and `_cleanup_daq_grpc` | yes — `collect_complete_filename` written but files remain on DAQ; next run has stale data |
| SC-030 | PH baseline file missing | no — `ph_baseline_file_ok()` catches |
| SC-031 | PH baseline file > 24 hours old | **yes** — **bug**: code uses `time.time() - 24*86400` (24 days!), not 24 hours. Test: set mtime to 26 h ago, expect refusal. Currently passes. |
| SC-032 | PH baseline file 0 bytes | yes — no size check today; driver reads zeros, MAROC mis-configured |
| SC-033 | Stale `tmp/interleave.pid` from a previous crashed run | yes — `stop_interleave()` reads PID and calls `os.kill(pid, SIGTERM)` without verifying process identity; could signal an unrelated PID, or silently fail and leave the file |
| SC-034 | Interleave daemon (`tools/interleave.py`, started via `config.py --start-interleave`) outlives `stop.py`'s `retry_limit=10 × 0.5 s = 5 s` | yes — `stop_interleave` gives up silently with no hard-kill fallback; daemon keeps flipping modes after stop.py completes |
| SC-034b | `config.py --start-interleave` invoked while interleave daemon is already running | yes — no advisory lock; two daemons race over quabo config |
| SC-035 | `quabo_uids.json` lists a UID the mock-quabo refuses | yes — `start_data_flow` calls `quabo.send_daq_params`; no-op on UDP, no error surfaced; data flow broken silently |
| SC-036 | Run directory collision (clock resolution = seconds) | yes — rapid sequential `start.py`/`stop.py` within same UTC second → `mkdir` raises, unclean abort |
| SC-037 | Head node IP mismatch (multi-homed box) | no — current code handles but pin it |
| SC-038 | `head_node_data_dir != data_dir` on same-IP head/DAQ | no — current code catches |
| SC-039 | `data_config.json` modified between `get_daq_params()` and `start_recording()` | yes — inconsistent state: quabos have mode A, hashpipe told mode B |
| SC-040 | `obs_config.json` timing_mode changes between session_start and start.py | yes — quabo 0 still running old WR firmware, inconsistent timing |

#### Data-plane integrity (SC-041 → SC-055)

| # | Scenario | TDD-forcing |
|---|---|---|
| SC-041 | tcpreplay with 5% packet loss | yes — packet_no gaps must be detected; current code has no gap accounting |
| SC-042 | tcpreplay with out-of-order packets | yes — PFF writer should tolerate; test pins behavior |
| SC-043 | Corrupt JSON header block (truncated before `\n\n`) | yes — `pff.py` parser behavior under truncation not specified |
| SC-044 | Binary image block without preceding `*` | yes — parser must error cleanly |
| SC-045 | Mixed `ph256` + `ph1024` frames in one stream (BOARDLOC inconsistent) | yes — current DAQ plugin may write to wrong file |
| SC-046 | Image mode 8-bit ↔ 16-bit switch at interleave boundary | yes — partial frames at the ~100 ms transition; tests assert the first ~100 ms of movie frames may have missing quabos (per docs) but that downstream code does not conflate them with persistent data loss |
| SC-047 | Movie mode + `two_pixel_trigger`/`three_pixel_trigger > 0` in same interleave state | no — Pydantic should reject at `config.py --validate`; pin it |
| SC-048 | Interleave state references undefined `movie_mode_config` or `pulse_height_mode_config` key (the key is not defined at the top level of `data_config.json`) | yes — `KeyError` with unhelpful trace |
| SC-048b | Interleave state sets *both* `movie_mode_config` and `pulse_height_mode_config` to `null` | no — Pydantic rejects per docs; pin it |
| SC-049 | `max_file_size_mb` rollover during interleave transition | yes — seqno continuity and **fixed JSON-header size invariant** across file boundary; PFF header size is set by the first frame, so a new file must re-establish its own frame[0] padding |
| SC-049b | JSON header length in frame *N* ≠ length in frame 0 of the same file | yes — breaks every mmap-strided reader (`PFFSequence`, etc.); no current asserter |
| SC-050 | Quabo slot 0 absent (empty UID), slot 1-3 present | yes — many places assume quabo 0 exists (timing_mode config is on quabo 0) |
| SC-051 | All 4 quabos of one module absent but module in `daq_config` | yes — `start_data_flow` skips each with empty UID; hashpipe listens for nothing; should log warning but continue |
| SC-052 | Module unreachable (quabo silent) | yes — no ping sweep before `start_data_flow`; discovered by absence of science packets minutes later |
| SC-053 | PFF file >1 GB (max_file_size_mb exceeded by 1%) | no — pin the rollover contract |
| SC-054 | Clock skew: \|tv_usec·10³ − pkt_nsec\| > 25 ms in either direction | yes — per `Precise-Timing.md`, timing utility must adjust `tv_sec ± 1` based on sign of `pkt_nsec − tv_usec·10³`; reference is `control/utils/pff.py` (line ~238). Test with synthetic packets at ±30 ms, ±500 ms; do **not** match against `panoseti_interface.py`'s 50 ms variant (unvetted). |
| SC-055 | `pkt_nsec` near UTC second boundary (`999_999_000` ns) with DAQ `tv_usec=1000` | yes — exercises the wrap-around correction in both `tv_sec + 1` and `tv_sec − 1` branches |

#### Telemetry and logging resilience (SC-056 → SC-068)

| # | Scenario | TDD-forcing |
|---|---|---|
| SC-056 | Loki container down for 30 s during run | yes — `storeLoki.py` should buffer or die loudly; current: silent log loss |
| SC-057 | Redis `maxmemory` reached | yes — `RedisBatcher` RPUSH fails; no backpressure |
| SC-058 | Telemetry gRPC server restarts mid-run | yes — AsyncGrpcHandler should reconnect; untested |
| SC-059 | Network partition `daqnode ↔ headnode` for 60 s | yes — telemetry logs lost; no local spool |
| SC-060 | storeLoki.py crashes on head | yes — `supervisord`-style restart? Currently: dead, logs pile in Redis |
| SC-061 | Log line 100 KB payload | yes — gRPC max-message-size default 4 MB, but Redis list entry size matters |
| SC-062 | Non-UTF8 bytes in log message | yes — Pydantic or JSON encoder explodes |
| SC-063 | 10k log/s burst for 5 s (hashpipe debug) | yes — batcher behavior under burst |
| SC-064 | Clock skew 2 s between head and DAQ | yes — Loki timestamps out-of-order → query gaps |
| SC-065 | `HEADNODE_IP` env var unset on daqnode | yes — `get_logger(grpc_enabled=True)` behavior undefined |
| SC-066 | Telemetry service down at daqnode startup | yes — startup should proceed, logs silently buffer |
| SC-067 | Redis connection drop during `RedisBatcher.flush()` | yes — 100-message batch lost? |
| SC-068 | `SANDBOX:` TTL expiry during a read | yes — race between writer and TTL |

#### Distributed orchestration (SC-069 → SC-080)

| # | Scenario | TDD-forcing |
|---|---|---|
| SC-069 | 3 DAQ nodes, node 2 drops during `StartDaq` | yes — partial-start, see SC-002 at scale |
| SC-070 | 3 DAQ nodes, node 1 drops during `StopDaq` | yes — node-0 stopped, node-2 never reached |
| SC-071 | 6 DAQ nodes, one with 200 ms latency | yes — sequential gRPC loop in `start_recording` serializes; measure total time exceeds sum of latencies |
| SC-072 | Rolling restart of DAQ nodes during active run | yes — run survives? Currently: no |
| SC-073 | `socat` gateway crashes during port-forwarded command | yes — one quabo unreachable, others fine |
| SC-074 | Module 128 moved from daqnode-1 to daqnode-2 between runs | yes — `quabo.data_packet_destination` sends to old IP; `daq_config.json` must be re-loaded |
| SC-075 | Head node = DAQ node (loopback) | no — pin current behavior |
| SC-076 | Head node separate from DAQ nodes | no — pin default |
| SC-077 | Two domes, different obs coords, same module IDs | yes — BOARDLOC uniqueness? |
| SC-078 | Port forwarding on some nodes, direct on others | yes — mixed topology code path |
| SC-079 | `module.config` write race between daqnode-1 and daqnode-2 (shared volume) | yes — already mitigated by `daq_data_2` volume; pin it as a regression test |
| SC-080 | `panoseti-server` unified server SIGHUP reload | yes — config reload behavior untested |

#### Storage topology (SC-S001 → SC-S010)

The DAQ-node multi-disk layout and head-node multi-volume layout (`Storage-on-DAQ-nodes.md`, `Storage-on-the-head-node.md`) are operationally critical but not exercised today. Each case below must set up a container with the disk topology under test via `docker-compose` volume overrides.

| # | Scenario | TDD-forcing |
|---|---|---|
| SC-S001 | `~panoseti/data/module_N/` is a symlink to `/mnt/diska/data/module_N/`; normal run completes | no — pin the symlinked-module-dir contract; assert PFF files land in the real target |
| SC-S002 | Symlink target (`/mnt/diska`) fills to 99% mid-run; main data volume has space | yes — ENOSPC visible only on the symlinked module; `start.py --validate-only` / `--disk_space` currently doesn't follow symlinks |
| SC-S003 | Symlink target read-only (simulate disk failure) | yes — hashpipe write errors surfaced how? currently: silent data loss on that module |
| SC-S004 | Dangling symlink: `module_N/` points at a missing path | yes — `make_run_dirs` creates sub-dir in the broken link → `OSError` mid-flight, no rollback |
| SC-S005 | Two module symlinks to the same disk; compute disk-space per underlying device, not per module | yes — naive summation over `df module_N` double-counts |
| SC-S006 | `head_node_data_dir` points at a volume with <1 GB free | yes — `collect.collect_data` must refuse *before* starting rsync, not partway through |
| SC-S007 | `head_node_data_dir` symlinked under `/home/panosetigraph/web/` correctly | no — pin the web-visibility contract |
| SC-S008 | Operator swaps `head_node_data_dir` between two runs to a volume that lacks the `data/analysis` sub-layout | yes — run should fail validation; currently `make_run_dirs` creates arbitrary structure |
| SC-S009 | Run started on volume A; `stop.py` invoked with `head_node_data_dir` swapped to volume B mid-flight | yes — `collect_data` looks in volume B, finds nothing, run data remains on DAQ |
| SC-S010 | Multiple modules on one DAQ node split across two disks via symlinks, concurrent write at full rate (~200 MB/s × 2) | no — throughput regression test; fails if disk-stripe assumption breaks |

#### Config validation (SC-081 → SC-094)

These extend `control/ci/unit/test_global_validator.py` but exercise the full `start.py --validate-only` path with mock-quabo responses where needed.

| # | Scenario | TDD-forcing |
|---|---|---|
| SC-081 | `integration_time_usec=7` (not multiple of 10) | no — pin contract |
| SC-082 | `integration_time_usec=7000` (doesn't divide 1e6) | no |
| SC-083 | `run_type="my run"` (space) | no |
| SC-084 | `run_type="verylongrunname01"` (>14 chars) | no |
| SC-085 | `detector_overvoltage` mismatch obs ↔ data | no |
| SC-086 | `pe_threshold=1.5` in PH mode | no |
| SC-087 | Interleave state with both movie-mode and `two_pixel_trigger`/`three_pixel_trigger > 0` | no — Pydantic rejects per the rule in `Interleaving-Observing-Mode-and-Configuration-Validation.md` |
| SC-087b | Interleave state with both `movie_mode_config=null` and `pulse_height_mode_config=null` | no — Pydantic rejects per the same docs |
| SC-088 | Interleave state `movie_mode_config`/`pulse_height_mode_config` references a top-level key that does not exist (or uses the reserved defaults wrongly) | yes — unhelpful `KeyError`; validator should name the missing key |
| SC-088b | Top-level extra mode key does not begin with `image_` or `pulse_height_` prefix | no — Pydantic rejects per docs; pin |
| SC-089 | Two quabos with same IP in obs_config | no |
| SC-090 | BOARDLOC collision (different domes, same module_id) | yes — validator misses |
| SC-091 | `module_ids` range overlap across daqnodes | yes — validator misses |
| SC-092 | WR firmware path missing in `wr/wrpc_filesys` | yes — silent failure at reboot |
| SC-093 | Firmware file listed but binary absent | yes — `config.py --loads` dies mid-flight |
| SC-094 | GNSS module configured with WR IP | yes — both try to claim port |

### 1.4 Fixture architecture

```
control/ci/integration/
├── conftest.py                    # existing — extended
├── chaos/
│   ├── __init__.py
│   ├── netem.py                  # tc qdisc wrappers
│   ├── iptables.py               # blackhole rules
│   ├── grpc_proxy.py             # in-process gRPC interceptor fixture
│   ├── process_chaos.py          # docker exec kill helpers
│   ├── disk_chaos.py             # fill/restore volumes
│   └── clock_chaos.py            # time skew
├── state_probe.py                 # state assertions
├── mock_quabo/                    # Dockerfile + server.py
│   ├── Dockerfile
│   ├── server.py
│   └── control_client.py         # pytest-side UDS client for fault injection
├── scenarios/
│   ├── test_sc_grpc_failures.py          # SC-001 .. SC-020
│   ├── test_sc_transactional_state.py    # SC-021 .. SC-040
│   ├── test_sc_data_integrity.py         # SC-041 .. SC-055
│   ├── test_sc_telemetry.py              # SC-056 .. SC-068
│   ├── test_sc_distributed.py            # SC-069 .. SC-080
│   └── test_sc_config_validation.py      # SC-081 .. SC-094
└── ...existing test files
```

Key fixtures (pseudocode):

```python
# ci/integration/conftest.py (additions)

@pytest.fixture
def mock_quabo_fleet(docker_client) -> MockQuaboFleet:
    """Yield a fleet handle; auto-resets state between tests."""
    fleet = MockQuaboFleet.attach("ctl-int-mock-quabo-1")
    try:
        fleet.reset_all()
        yield fleet
    finally:
        fleet.reset_all()

@pytest.fixture
def state_probe(daq_control_client, redis_client, loki_client) -> StateProbe:
    return StateProbe(daq_control_client, redis_client, loki_client)

@pytest.fixture
def netem(request):
    """Apply tc netem to a daqnode egress for test duration."""
    ctx = NetemContext()
    yield ctx
    ctx.restore_all()

@pytest.fixture
def grpc_proxy(request):
    """In-process gRPC interceptor (channel-level monkey-patch)."""
    proxy = GrpcChaosProxy()
    yield proxy
    proxy.restore()

@pytest.fixture
def kill_hashpipe(docker_client):
    """Returns a callable (node_name, delay_s) -> None."""
    killers = []
    def _kill(node, delay=0, signal="KILL"):
        killers.append(spawn_killer(docker_client, node, delay, signal))
    yield _kill
    for k in killers: k.cancel()

@pytest.fixture
def fresh_run_state(daq_config):
    """Pre-test: clear current_run, any stale PIDs, kill rogue hashpipe."""
    reset_observatory_state(daq_config)
    yield
    reset_observatory_state(daq_config)

@pytest.fixture
def orphaned_hashpipe(kill_hashpipe, daq_control_client):
    """Start hashpipe, SIGKILL it, leave server with stale PID."""
    daq_control_client.StartDaq(default_start_args())
    kill_hashpipe("daqnode", signal="KILL")
    wait_for_process_death("daqnode", "hashpipe", timeout=5)
    yield  # pid > 0 in server memory, process dead
```

### 1.5 TDD-forcing test exemplars (proof of current brokenness)

These six concrete tests should each **fail red** on current `master`. Implementation order matches TDD priority.

#### Exemplar A — SC-010: Orphaned hashpipe blocks cleanup forever (requires `--force` to delete data)

Per Appendix B-2: even after liveness detection improves, `CleanupData` must **refuse** to delete non-empty module dirs unless the caller sets `force=true`. This prevents accidental destruction of science data.

```python
def test_SC010_orphaned_hashpipe_blocks_cleanup_without_force(
    mock_quabo_fleet, kill_hashpipe, state_probe, fresh_run_state
):
    run_name = invoke_start_py()
    assert state_probe.hashpipe_process_alive("daqnode")

    # Simulate hashpipe crash (SIGKILL — no clean exit, no pid cleanup)
    kill_hashpipe("daqnode", signal="KILL")
    wait_for_process_death("daqnode", "hashpipe", timeout=5)

    # stop.py without --force-cleanup: must preserve data
    with pytest.raises(CleanupRefusedPreserveData) as exc:
        invoke_stop_py()  # plain stop, no force
    assert "FAILED_PRECONDITION" in str(exc.value)
    assert any_pff_files_on_daqnode(run_name), \
        "Run data was deleted without an explicit --force-cleanup"

    # With --force-cleanup: data is removed AND incident key written
    invoke_stop_py(force_cleanup=True)
    assert state_probe.current_run() is None
    assert not any_pff_files_on_daqnode(run_name)
    assert state_probe.redis_incident_key(f"panoseti:incident:forced_cleanup:{run_name}")
```

**Why this fails today:** (a) `panoseti_grpc/daq_control/server.py` gates `CleanupData` on `self.hashpipe_pid > 0` without verifying `/proc/<pid>` exists — the orphan case is currently stuck forever with no escape hatch. (b) There is no `force` field on `CleanupDataRequest` and no corresponding `stop.py --force-cleanup` flag; the proto must be extended.

> **Companion case SC-010b** — `CleanupData(force=true)` on a *live* hashpipe must still be refused (server returns `FAILED_PRECONDITION`). `force` is only an override for the dead-PID path, not a global kill switch.

#### Exemplar B — SC-002: Partial start corrupts array

```python
@pytest.mark.parametrize("failing_node_idx", [0, 1])
def test_SC002_partial_start_rolls_back_all_nodes(
    grpc_proxy, mock_quabo_fleet, state_probe, fresh_run_state,
    failing_node_idx
):
    # Inject UNAVAILABLE on one DAQ node
    grpc_proxy.set_mode(
        f"daqnode-{failing_node_idx}", "StartDaq", "UNAVAILABLE"
    )
    with pytest.raises(StartRunFailed):
        invoke_start_py()

    # ASSERTION: no partial *active* state anywhere
    assert state_probe.current_run() is None
    assert not state_probe.hashpipe_process_alive("daqnode")
    assert not state_probe.hashpipe_process_alive("daqnode-2")
    for quabo in mock_quabo_fleet.all():
        # Quabo returned to post-session_start steady state (no data destination)
        assert quabo.state.data_dest is None, \
            f"quabo {quabo.ip} still pointed at DAQ node after failed start"
    assert not state_probe.hk_recorder_running()
    assert not state_probe.hv_updater_running()

    # ASSERTION (per Appendix B-1): partial artifacts are preserved for post-mortem
    aborted_root = state_probe.aborted_snapshot_root()   # <head_node_data_dir>/_aborted/
    snapshots = list(aborted_root.iterdir())
    assert len(snapshots) == 1, "expected exactly one aborted-run snapshot"
    snap = snapshots[0]
    assert (snap / "start_failure_context.json").exists(), \
        "no failure context captured for debugging"
    # Snapshot must include whatever module_* and head-node dirs were created before failure
    assert any(snap.rglob("module_*")), "no partial module dirs captured"
```

**Why this fails today:** `start_run()` calls `start_data_flow()` *before* `start_recording()`. When `StartDaq` fails, quabos are already configured with packet destinations; there's no `stop_data_flow()` rollback, no `kill_hv_updater()` cleanup, and no post-mortem snapshot of the partial run directory.

#### Exemplar C — SC-006: StopDaq partial failure leaves zombie hashpipes

```python
def test_SC006_stop_continues_after_per_node_failures(
    grpc_proxy, state_probe, fresh_run_state, mock_quabo_fleet
):
    invoke_start_py()
    assert state_probe.hashpipe_process_alive("daqnode")
    assert state_probe.hashpipe_process_alive("daqnode-2")

    # Node 0's StopDaq times out
    grpc_proxy.set_mode("daqnode", "StopDaq", "timeout", timeout_s=45)

    with pytest.raises(StopPartialFailure):
        invoke_stop_py()

    # ASSERTION: node-2 still got a StopDaq even though node-0 failed
    assert not state_probe.hashpipe_process_alive("daqnode-2"), \
        "Node-2 was never told to stop because node-0 raised first"
    # Head should surface the error clearly
    assert state_probe.stop_errors_log_contains("daqnode timeout")
```

**Why this fails today:** `stop.py::stop_recording` raises Exception on the first failed `StopDaq`, skipping subsequent nodes.

#### Exemplar D — SC-031: PH baseline staleness off-by-24×

```python
def test_SC031_ph_baseline_24h_limit_not_24days(fresh_run_state):
    ph_file = Path("configs/quabo_ph_baseline.json")
    ph_file.write_text('{"quabos":[]}')
    # Set mtime to 26 hours ago (should be rejected)
    os.utime(ph_file, (time.time() - 26*3600, time.time() - 26*3600))

    with pytest.raises(PHBaselineTooOld):
        invoke_start_py()
```

**Why this fails today:** `start.py::ph_baseline_file_ok` compares against `time.time() - 24*86400` (24 days). Test file is only 26 hours old and passes.

#### Exemplar E — SC-024: Concurrent start corruption

```python
async def test_SC024_concurrent_start_only_one_wins(fresh_run_state, state_probe):
    async with asyncio.TaskGroup() as tg:
        t1 = tg.create_task(invoke_start_py_async())
        t2 = tg.create_task(invoke_start_py_async())

    outcomes = [t1.result(exception=True), t2.result(exception=True)]
    # Exactly one succeeds; the other gets clear "run in progress" error
    winners = [o for o in outcomes if o is None]
    losers  = [o for o in outcomes if isinstance(o, RunAlreadyInProgress)]
    assert len(winners) == 1 and len(losers) == 1
    # No double-start of hashpipe
    assert state_probe.hashpipe_pid("daqnode") is not None
    assert count_hashpipe_procs("daqnode") == 1
```

**Why this fails today:** No lockfile around `start_run`; both invocations pass `read_run_name()` check, both call `make_run_dirs`, one dies with `FileExistsError`, state is mixed.

#### Exemplar F — SC-033: Stale interleave PID from unrelated process

```python
def test_SC033_stale_interleave_pid_detected(fresh_run_state, tmp_path):
    # Seed a PID file with PID of an unrelated process (e.g. PID 1)
    Path("tmp/interleave.pid").write_text("1\n")

    # stop_interleave should not assume PID 1 is ours
    invoke_stop_py()  # should not raise, should detect and clean
    assert not Path("tmp/interleave.pid").exists()
    # PID 1 still alive (we didn't kill init)
    assert Path("/proc/1").exists()
```

**Why this fails today:** `stop_interleave()` reads the PID and does `os.kill(pid, SIGTERM)` without verifying the process is our interleave. Could kill unrelated processes, or silently fail on permission error leaving file.

### 1.6 TDD implementation order

Mid-level engineer work plan:

1. **Week 1 — Infrastructure:** `mock_quabo` service, `state_probe`, basic chaos primitives (`process_chaos`, `disk_chaos`).
2. **Week 2 — Exemplars A–F:** Six TDD-forcing tests failing red on `master`. No production code changes yet.
3. **Week 3 — Rewrite `start.py` / `stop.py`** with transactional design (context-manager-based rollback ladder). Exemplars A–F turn green.
4. **Week 4 — Fill in SC-001 → SC-040:** gRPC + state corruption matrix.
5. **Week 5 — SC-041 → SC-068:** data plane + telemetry.
6. **Week 6 — SC-069 → SC-094:** distributed + config.
7. **Ongoing:** each new bug discovered in ops gets a new SC-### before fix.

---

## Pillar 2 — Hardware-in-the-Loop (HITL)

### 2.1 Environment

HITL reuses `ci/Dockerfile.ci` **exactly** but with a modified `docker-compose.hitl.yml`:

- Removes the `mock-quabo` service.
- Replaces `tcpreplay` injection with real quabo UDP traffic.
- Binds daqnode container to host network (`network_mode: host`) on the mountain subnet so real quabos and WR switch are reachable.
- Requires ops-time flag: `RUN_REAL_HARDWARE=1`.
- Head node runs natively (not containerized) on the observatory host; DAQ nodes may be containerized if the host kernel supports `TPACKET_V3` on the physical NIC (which it does in production, unlike Docker virtual NICs).

### 2.2 Test taxonomy

Tests numbered `HW-###`.

#### Full lifecycle (HW-001 → HW-010)

| # | Scenario |
|---|---|
| HW-001 | Cold start: `session_start.py --no_hv` → `start.py --nsecs 60` → `stop.py` → `session_stop.py`. Assert PFF files >0 bytes for all 4 quabos × all modules, HK file has >15 packets per quabo, sw_info.json written, Redis has recent WR/HK keys. |
| HW-002 | Warm restart: back-to-back runs without session_stop. Assert that quabo configuration persists and `get_uids.py` returns same UIDs. |
| HW-003 | Full HV lifecycle: `--hv_on` → `start.py` → `stop.py` → `--hv_off`. Assert HVMON readings cross threshold during run, return to 0 after off. |
| HW-004 | Maroc calibration refresh between runs. |
| HW-005 | PH baseline recalibration and confirmation via `--show_ph_baselines`. |
| HW-006 | Two-dome observation simultaneously (if available). |
| HW-007 | GNSS timing mode end-to-end with NANOSEC reconstruction. |
| HW-008 | WR timing mode end-to-end with NANOSEC reconstruction. |
| HW-009 | Interleave schedule: image_8bit (2 s) ↔ pulse_height_uhe (58 s) for 10 cycles. Verify no frame loss at boundaries. |
| HW-010 | `--dry-run-interleave` does not touch hardware (verify HVMON unchanged). |

#### Rapid parameter switching (HW-011 → HW-020)

| # | Scenario |
|---|---|
| HW-011 | Run with 8-bit img (20 µs) → stop → run with 16-bit img (100 µs). |
| HW-012 | Change detector overvoltage 2 → 3 V between runs. |
| HW-013 | Toggle `group_ph_frames` between runs; assert PH file naming changes `ph256` ↔ `ph1024`. |
| HW-014 | Change `max_file_size_mb` from 1000 → 100 mid-session; next run rotates faster. |
| HW-015 | Enable 2-pixel trigger between runs; assert GOEMASK correctly propagates. |
| HW-016 | Stim pulse ON → OFF: verify artificial rate appears/disappears. |
| HW-017 | Flash LED ON during 8-bit image; correlate packet timestamps with 1PPS. |
| HW-018 | Reconfigure channel mask mid-session via `--mask_config`. |
| HW-019 | Change `integration_time_usec` in interleave `states` without restart. |
| HW-020 | Swap `obs_config.json` symlink to a variant with different module count. |

#### Hardware degradation (HW-021 → HW-030)

| # | Scenario |
|---|---|
| HW-021 | Unplug one quabo Ethernet cable mid-run. Expect: no crash, missing quabo silent in PFF, HK packets stop for that BOARDLOC, logged at WARN. |
| HW-022 | Power-cycle quabo between runs; confirm UIDs match post-reboot. |
| HW-023 | WR switch drops 10MHz briefly (`EXT_10MHz_STATUS=0` in HK). Assert `EXT_1PPS_STATUS=0` as invalidation cascade. |
| HW-024 | GPS antenna unplugged → `capture_gps.py` continues, NaN values in Redis. |
| HW-025 | DAQ node disk fills to 99% during long run. Expect: graceful stop, partial data collected. |
| HW-026 | Head-node data volume unmounts (simulated). Expect: next run refuses start, clear error. |
| HW-027 | Web Power Switch unreachable at `session_start`. Expect: specific error, no partial state. |
| HW-028 | 1 quabo with burned detector (HVMON saturated). Expect: detection and skip. |
| HW-029 | Network latency spike (physical cable swap) during run. |
| HW-030 | Operator ctrl-C during `session_start`. Assert no half-configured modules. |

### 2.3 HITL gating strategy

- HITL runs nightly via cron on observatory head node.
- Results posted to Loki + Slack channel.
- Deploy tag `rN.N` requires 7 consecutive clean nightly HITL runs.
- Mountain engineer can kick ad-hoc HITL via `pseti grpc status`.

---

## Pillar 3 — Docker Compose Evolution (N DAQ nodes)

### 3.1 Design constraints

- Must support 2 (default), 4, 6, and configurable N nodes.
- Each node needs: unique IP on `daqnode_net`, unique volume (avoid `module.config` write races), unique container name.
- Must run on laptop (N≤4) and on CI runner (N≤8).
- Must not regress current 2-node tests.

### 3.2 Recommended approach — pytest-driven parametric topology

Keep `docker-compose.integration.yml` for the fixed infrastructure (redis, loki, gateway, headnode, int-tester). **Remove** the explicit `daqnode` and `daqnode-2` entries. Instead:

1. A pytest fixture `daqnode_fleet(n)` uses the `docker` Python SDK to instantiate N daqnode containers from the `integration-daqnode` image, assigning IPs and volumes dynamically.

```python
# ci/integration/fleet.py

MAX_DEFAULT_FLEET_N = 4           # N > 4 requires RUN_LARGE_FLEET=1
DAQNODE_SHM_BYTES = 2 * 1024**3   # hashpipe needs ≥2 GB /dev/shm per container

@dataclass
class DaqnodeSpec:
    name: str
    ip: str
    volume_name: str
    module_ids: list[int]

@pytest.fixture
def daqnode_fleet(request, docker_client):
    """Usage: @pytest.mark.parametrize('daqnode_fleet', [2,4], indirect=True)
    Higher N requires RUN_LARGE_FLEET=1 in the environment.
    """
    n = request.param
    if n > MAX_DEFAULT_FLEET_N and not os.getenv("RUN_LARGE_FLEET"):
        pytest.skip(f"N={n} requires RUN_LARGE_FLEET=1 (ensures sufficient shm)")

    image = ensure_image_built("integration-daqnode")
    specs = [
        DaqnodeSpec(
            name=f"ctl-int-daqnode-dyn-{i}",
            ip=f"192.168.0.{10 + i}",
            volume_name=f"daq_data_dyn_{i}",
            module_ids=module_id_slice(i, n),
        )
        for i in range(n)
    ]
    # Fleet.start() passes shm_size=DAQNODE_SHM_BYTES and checks
    # `df -B1 /dev/shm` ≥ DAQNODE_SHM_BYTES per container before yielding.
    fleet = Fleet(docker_client, specs, shm_bytes=DAQNODE_SHM_BYTES)
    fleet.start()
    fleet.wait_healthy(timeout=60)
    fleet.verify_shm()  # fail-fast if /dev/shm is undersized
    try:
        yield fleet
    finally:
        fleet.stop_and_remove()
```

2. `Fleet.start()` generates a matching `daq_config.json` on the `int-tester` filesystem and writes it to a test-scoped config dir the tests point at via `PANOSETI_CONFIG_DIR`.

3. Tests that care about N opt in:

```python
@pytest.mark.parametrize("daqnode_fleet", [2, 4], indirect=True)
def test_SC071_startdaq_latency_scales(daqnode_fleet, state_probe):
    t0 = time.monotonic()
    invoke_start_py()
    elapsed = time.monotonic() - t0
    assert elapsed < 5 + 0.5 * daqnode_fleet.n_nodes
```

4. Existing 2-node tests continue to use the compose-defined `daqnode` and `daqnode-2` — no regression.

### 3.3 Why not `deploy.replicas` or Jinja2?

- `deploy.replicas` doesn't support per-replica IPs/volumes without Swarm.
- Jinja2-rendered compose files work but fork the compose surface area; two sources of truth for topology.
- Dynamic Python SDK approach lets tests assert against topology-as-data, not YAML.

### 3.4 Scaling experiments (Pillar 3 specific tests)

Tests numbered `SC-N###` (scaling-N).

| # | Scenario |
|---|---|
| SC-N001 | Sequential `StartDaq` to N∈{2,4} nodes — measure total wall time, assert it scales linearly with N (exposes missing parallelism). |
| SC-N002 | Parallel `StartDaq` via `asyncio.gather` on N∈{2,4} — assert total time ≈ slowest node (forces refactor of `start_recording` loop). |
| SC-N003 | N=4, kill node 2 mid-start. Assert rollback of nodes 0–1 and that node 3 is never issued `StartDaq`. |
| SC-N004 | N=4, simulate 300 ms latency on node 2. Assert start does not block other nodes. |
| SC-N005 | N=4, all fail. Assert clear aggregate error message. |
| SC-N006 | N=2 vs N=4 — compare telemetry volume, assert `RedisBatcher` keeps up. |
| SC-N007 *(opt, `RUN_LARGE_FLEET=1`)* | N=8 stress run on a dedicated runner — all-pass smoke + aggregate shm usage ≤ 16 GB. |

---

## File & directory layout (complete)

```
control/
├── ci/
│   ├── Dockerfile.ci                    # unchanged (mock_quabo stage added)
│   ├── docker-compose.integration.yml   # daqnode/daqnode-2 entries REMOVED; services: redis, loki, gateway, headnode, int-tester, mock-quabo
│   ├── docker-compose.hitl.yml          # NEW: host-net variant
│   ├── qa.toml                          # add `[test.chaos]`, `[test.hitl]`
│   ├── qa.py                            # add `chaos`, `hitl` commands
│   ├── README.md                        # document mock_quabo + dynamic fleet
│   └── integration/
│       ├── conftest.py                  # extended
│       ├── chaos/                       # NEW
│       ├── state_probe.py               # NEW
│       ├── mock_quabo/                  # NEW
│       ├── fleet.py                     # NEW (dynamic topology)
│       ├── scenarios/                   # NEW (SC-### matrix)
│       ├── hitl/                        # NEW (HW-### matrix)
│       └── build/                       # existing
├── start.py                             # needs transactional rewrite (driven by tests)
├── stop.py                              # needs rewrite (partial-failure tolerant)
└── config.py                            # PH baseline 24h fix
```

---

## Verification strategy

End-to-end proof the suite works, in order:

1. **Red baseline:** run exemplars A–F on current `master`. All six must fail for the documented reasons. If any pass, the injection is weak — rework before proceeding.
   ```bash
   python ci/qa.py chaos -k "SC010 or SC002 or SC006 or SC031 or SC024 or SC033"
   # expect: 6 failed, 0 passed
   ```
2. **Production fix PR per exemplar:** TDD rewrite of `start.py`/`stop.py`/`daq_control/server.py`. Merge only when the 6 exemplars go green and all existing `ci/unit/` + `ci/integration/` tests still pass.
3. **Full SC matrix green:** after rewrite, `python ci/qa.py chaos` passes all 94+ cases.
4. **Dynamic fleet validated:** `python ci/qa.py chaos -k "SC-N"` passes for N ∈ {2, 4, 6}.
5. **HITL smoke:** on mountain, `python ci/qa.py hitl -k HW-001` passes end-to-end with real hardware.
6. **HITL full:** nightly cron running `python ci/qa.py hitl` for 7 days without regression → tag release.
7. **Regression gate:** every subsequent ops incident produces a new `SC-###` before the fix merges.

---

## Appendix A — TDD-forcing bug inventory (recap)

| SC# | File:line | Bug summary |
|---|---|---|
| SC-002 | `start.py:start_run` | No rollback ladder between `make_run_dirs`, `start_data_flow`, `start_recording`. |
| SC-006 | `stop.py:stop_recording` | First-failure raise skips remaining DAQ nodes. |
| SC-010 | `panoseti_grpc/daq_control/server.py` | `hashpipe_pid > 0` gate on CleanupData lacks liveness check. |
| SC-024 | `start.py:start_run` | No advisory lock; concurrent starts race on `read_run_name()`. |
| SC-027 | `stop.py:stop_run` | `--run X` ignores `current_run=Y` mismatch. |
| SC-031 | `start.py:ph_baseline_file_ok` | `time.time() - 24*86400` is 24 days, comment says 24 hours. |
| SC-033 | `stop.py:stop_interleave` | `os.kill(pid)` without verifying process identity. |
| SC-034 | `stop.py:stop_interleave` | `retry_limit=10` × 0.5 s = 5 s; interleave can legitimately take longer; no hard-kill fallback. |
| SC-035 | `start.py:start_data_flow` | No ping-sweep before configuring quabos; silent failure on unreachable quabo. |
| SC-039 | `start.py` | Config loaded once; modification between `get_daq_params` and `start_recording` isn't re-read. |
| SC-056 | `daemons/storeLoki.py` | No local spool; Loki outage = lost logs. |
| SC-060 | daemons | `storeLoki.py` crash has no supervisor. |
| SC-067 | `panoseti_grpc/telemetry/server.py:RedisBatcher` | Connection drop during flush loses 100-msg batch. |
| SC-071 | `start.py:start_recording` | Sequential gRPC loop — latency scales linearly in N. |

## Appendix B — Resolved design decisions (from stakeholder review)

1. **Rollback semantics for `start.py` mid-flight failure — RESOLVED.**
   The rollback ladder restores the observatory to the **powered-on + hardware-configured** state (i.e. the post-`session_start.py` steady state). All partially-created distributed observing-run artifacts are torn down: run directories on head node and every DAQ node, `current_run` marker, Redis run keys, quabo data-packet destinations.
   *Caveat (must be implemented):* the rollback must **preserve a post-mortem snapshot** of the partial files and any error context in a sibling directory (e.g. `<head_node_data_dir>/_aborted/<run_name>/`) so an operator can debug the partial failure. A `--keep-artifacts` flag on `start.py` (or equivalent metadata) controls whether the snapshot stays indefinitely or is pruned by a housekeeping job. Tests must assert both: (a) the active run tree is gone; (b) the aborted snapshot exists and is readable.

2. **Orphaned hashpipe policy — RESOLVED.**
   `CleanupData` must **refuse to delete run data** when it detects a dead hashpipe PID with non-empty `module_*` dirs, unless an explicit `force=true` field is set on the `CleanupDataRequest` (propagated from a `stop.py --force-cleanup` operator flag). Rationale: we must never accidentally and permanently delete science data. Server contract:
   - `CleanupData` without `force`: server performs the liveness check, and if the PID is dead but data exists, returns `FAILED_PRECONDITION` with a descriptive message that includes the remaining byte count per module. Tests must pin this contract (new case **SC-010b**).
   - `CleanupData` with `force=true`: server deletes, logs at `ERROR`, and writes a Redis incident key `panoseti:incident:forced_cleanup:<run>` with timestamp, operator hostname, and module byte counts.
   - Tests must assert neither a crashed hashpipe nor a transient network glitch can invoke the destructive path without the flag.

3. **Interleave hard-kill after `retry_limit` — RESOLVED (current approach).**
   Use the recommended path for the initial refactor: `SIGKILL` the `tools/interleave.py` process if it outlives the 5 s retry window, then call `config.py --maroc_config` synchronously to restore the default MAROC registers. Rationale: minimize blast radius while the control plane is still Python/UDS-based.
   *Future direction (out of scope for this test suite, but referenced so tests don't lock in the current architecture):* interleave execution will move out of the head-node Python process and into a subprocess managed by the **DAQ Control server** on each DAQ node, with interleave schedules shipped as part of the `StartDaq` payload or a new `StartInterleave` RPC. Until that refactor lands, tests target the current `tools/interleave.py` daemon and its PID file at `tmp/interleave.pid`. When the refactor starts, the `stop_interleave` tests become RPC-level tests against the DAQ Control server instead.

4. **HITL cadence — TBD.**
   Deferred pending operations team input. Design assumption: no HITL gate on PRs yet; HITL results publish to Loki + (optional) Slack; merge-gating on HITL is a future flip.

5. **Fleet size for CI — RESOLVED.**
   `N = 4` DAQ nodes in the `daqnode_fleet` scaling fixture. Each DAQ container must be provisioned with **≥ 2 GB of shared-memory (`/dev/shm`)** to satisfy hashpipe's shared-memory buffers. Fleet fixture must set `shm_size="2g"` (Docker SDK `host_config`) on every daqnode container it creates. A CI fail-fast pre-flight check verifies `df -h /dev/shm` ≥ 2 GB per container before any test runs. Scaling tests (`SC-N001`–`SC-N006`) parametrize over `{2, 4}` only; `N=6`/`N=8` references in the scaling matrix are demoted to optional runs gated on `RUN_LARGE_FLEET=1`.

6. **Mock-quabo coverage — RESOLVED.**
   `mock_quabo` implements **only the packet interface** (`Quabo-packet-interface.md`). It does not model firmware bugs or hardware errata. The software-only tests assume hardware behaves ideally; this keeps pass/fail signal attributable to control-plane code, not to simulated hardware flakiness.
   *Science-data generation:* the existing `tcpreplay`-from-PCAP pattern in `docker-compose.integration.yml` stays as the baseline for streaming science UDP packets. Where a test needs a handful of deterministic science packets (e.g. timing-boundary SC-054/SC-055), a lightweight asyncio UDP packet generator inside `mock_quabo` is acceptable — crafting a few UDP datagrams on the fly is simpler than maintaining more PCAPs for one-off cases. Any such generator lives behind a control-UDS command (`emit_science_packet {header...}`) so tests can drive it without restarting the container.
