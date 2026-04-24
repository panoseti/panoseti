# PanoSETI Architecture Modernization & Chaos Testing Blueprint

## Context

The observatory control plane has grown into four production gRPC services (`daq_data`, `daq_control`, `telemetry`, `ublox_control`) wired together by an async head-node orchestrator. Recent work (Py3.14 migration, `StartTransaction`/`StopTransaction` context managers, a decoupled `transfer_daemon`, a mature `chaos/` harness, and a new `pseti test hw` CLI) has moved the system forward considerably. What's missing is:

1. **Shared client machinery** — each service has re-implemented channel lifecycle, error wrapping, health probes, and sync/async method pairs. This is now the single largest source of drift and latent bugs.
2. **Standard protocols** where we've rolled our own — no `grpc.health.v1`, no interceptors, no service-config retry policy.
3. **End-to-end transactional integrity** — manifest generation works, but the `VERIFYING` stage in the transfer daemon is a stub (`utils/transfer/daemon.py:153`). `verify.py` exists and is unwired. We can in principle delete DAQ-side data without ever having proved head-node-side integrity.
4. **Chaos + HW coverage** — the harness is ready; failure-mode scenarios for the transfer daemon and end-to-end HW tests beyond `test_01_environment.py` have not been written yet.

This plan lays out (a) a modernization blueprint for the gRPC layer, (b) a component-by-component evaluation of DaqData/Telemetry, (c) a transactional-integrity fix plus a chaos + HW test suite. **No implementation code in this document.**

---

## Phase 1: gRPC Architecture & Modernization

### 1.1 Shared `grpc_utils` package

Create `src/panoseti_grpc/grpc_utils/` inside the grpc submodule. It holds *client-and-server-agnostic* machinery:

| Module | Responsibility | Replaces |
|---|---|---|
| `channel.py` | `AsyncChannelFactory` + `MultiHostChannelManager` — owns channel creation, keepalives, per-host stub cache, async `__aenter__`/`__aexit__` with TaskGroup-based fan-in, and a single `service_config` JSON applying method-level retry/hedging. | Duplicated `self.hosts = {h: {"channel","stub"}}` pattern in `daq_data/client.py:160-198`, plus the commented-out retry block in `telemetry/client.py:56-75`. |
| `exceptions.py` | `PanosetiRpcError` base + typed subclasses (`UnavailableError`, `DeadlineExceededError`, `ResourceExhaustedError`, `FailedPreconditionError`, …). Preserves original `grpc.RpcError`, `.code()`, `.details()`, target host, and request_id. | 12+ hand-written `except grpc.RpcError: raise ConnectionError(f"gRPC failed: {e.details()}") from e` blocks in `daq_control/client.py` alone. Callers (`start.py:567`, `stop.py:330`) stop unwrapping `e.__cause__`. |
| `decorators.py` | `@rpc_method` (async + sync variants) — validates input via an accompanying Pydantic model, auto-injects deadline and request_id, maps `grpc.RpcError → PanosetiRpcError`, emits structured telemetry. Pairs with the existing server-side `util/error_handling.py::grpc_error_handler`. | Per-method sync/async duplication in `daq_control/client.py:59-355` (~500 LOC collapses to ~150). |
| `interceptors.py` | Client interceptors (`LoggingInterceptor`, `DeadlineInterceptor`, `MetadataInterceptor`); server interceptors (`ExceptionInterceptor` replacing per-RPC `@grpc_error_handler`, `AccessLogInterceptor`). | No interceptors exist today (zero grep hits). |
| `retries.py` | Builder for the gRPC `service_config` retry policy JSON (declarative, transport-level). Wired once in `channel.py`. | Hand-rolled retry loop in `start.py:550-576` (`UNAVAILABLE`-only, fixed 1 s sleep). |
| `health.py` | Thin wrapper over `grpc_health.v1` — `register_health(server, services)` on the server side, `HealthClient.check(service, timeout)` / `watch()` on the client side. | `daq_data.Ping` RPC (`daq_data_pb2_grpc.py:80`, `server.py:247`), `daq_control`'s ad-hoc reuse of `StatusDaq` as a liveness probe (`start.py:594-647`). |
| `clients/` | Thin generated-stub wrappers living here rather than under each service. Service-specific business methods stay in `<service>/client.py`; channel management and RPC wrapping do not. | Channel + error-wrapping boilerplate across `daq_data`, `daq_control`, and `telemetry`. (`ublox_control` excluded from this refactor.) |

**Placement rule:** `grpc_utils/` depends on `generated/`, `panoseti_util/`, and nothing else. It is importable from `control/` so the orchestrator can share the same `MultiHostChannelManager`.

### 1.2 Asyncio modernization — `asyncio.TaskGroup`

TaskGroup is already the dominant pattern (`start.py:163, 587, 650, 658, 743`; `stop.py:505`; `utils/transfer/daemon.py:122, 180`). Remaining work is targeted:

**Migration targets:**

| Site | Current | Correct pattern | Rationale |
|---|---|---|---|
| `stop.py:361` | `asyncio.gather(*(stop_node(n) for n in daq_nodes))` without `return_exceptions` | Each `stop_node` wraps its body in `try/except` returning a `NodeOutcome(host, ok, error)`; driver uses `TaskGroup` and reduces over outcomes. | Current code intends best-effort ("Best-effort stop of all remote DAQ nodes") but one failure cancels siblings. |
| `start.py:725` | `asyncio.gather(*tasks)` for Quabo reachability | Same outcome-collection pattern under `TaskGroup`. | Same reason. |
| `AioDaqDataClient.__aexit__` (`daq_data/client.py:635-644`) | Suppresses `asyncio.CancelledError`, returns `True` | Re-raise `CancelledError`; convert only `grpc.RpcError` to `PanosetiRpcError`. | Breaks cooperative cancellation contract; masks shutdown hangs. |
| `stop.py:347` | `loop.run_in_executor(None, lambda: subprocess.run(...))` | `await asyncio.create_subprocess_exec(...)` (matches `start.py:154`) | The subprocess pattern is already correct elsewhere. |
| `telemetry/client.py` (sync + `threading.Thread` + `queue.Queue`) | Mixed threading model | `grpc.aio` + bounded `asyncio.Queue` | Only threaded code left in the stack. Blocked until the Telemetry Log-path decision in §2.2 is resolved. |

**Decision framework** (document in `grpc_utils/README.md`):
- **`TaskGroup`** → all-or-nothing fan-outs (startup sequence, teardown rollback ladder, manifest generation, stream merge on failure). First raise cancels siblings; `ExceptionGroup` surfaces at `__aexit__`.
- **Outcome-collection + `TaskGroup`** → best-effort fan-outs (status probes, cleanup attempts, rollback stop-all). Each task captures its own exception; driver sees no raises and reduces.
- **Never** `asyncio.gather(..., return_exceptions=True)` as an error-swallower (current use at `daq_data/client.py:864` discards the exception). If you need it for channel cleanup, log the exceptions.

### 1.3 Health checks — adopt `grpc.health.v1`

Register one `HealthServicer` on the unified `PanosetiServer`. For each registered service, expose the service name from `ServiceDescriptor` and default-mark it `SERVING` post-start. Services can flip to `NOT_SERVING` during reconfiguration (e.g., `daq_data.InitHpIo` while holding `_writer_lock`, `daq_control.StartDaq` mid-launch).

**Client-side benefits:**
- `HealthClient.check("panoseti.daq_control")` replaces the `StatusDaq`-as-heartbeat pattern in `start.py:594-647` (keeping `StatusDaq` for what it should be — hashpipe PID + disk usage).
- `HealthClient.watch(...)` delivers reactive updates; the orchestrator can cancel ongoing work the moment a node goes `NOT_SERVING` instead of polling.
- Compatible with `grpc_health_probe` (CLI) so the `pseti-grpc status` command and the `hardware-software/test_01_environment.py::test_grpc_liveness` check collapse to one standard call.

**Migration:**
1. Add `grpcio-health-checking` to `pyproject.toml`.
2. Implement `grpc_utils.health.register_health(server, toggles)` in PanosetiServer's `serve()`.
3. Add client-side `HealthClient` in `grpc_utils/health.py`.
4. Convert `daq_data`'s custom `Ping` RPC: leave the generated stub in place for one release with a `DeprecationWarning`; after the CLI and tests migrate, remove the proto RPC.

---

## Phase 2: Component Evaluation

### 2.1 DaqData — keep gRPC streaming, formalize the cache

**Current reality:** `latest_data_cache` is a shared single-slot dict keyed by `(module_id, "ph"|"movie")` (`hp_io_manager.py:52`). Each reader owns a cursor (`ReaderState`, `state.py:74-75`) and polls at its own `update_interval_seconds`. Fast producers overwrite slow consumers — **lossy by design**. Production consumer count today: **zero** (grep for `stream_images` / `DaqDataClient` shows only tests, CLI plotting, and a demo notebook).

**Recommendation: KEEP the current design.** Arguments:

| Option | For | Against |
|---|---|---|
| **Stay with gRPC streams** | Server-side per-reader rate-shaping + cursor is the *correct* model for low-bandwidth previews of an overwrite-tolerant stream. No new dependency. Zero consumers today — re-architecting is premature. | `StreamImages` polls rather than awaits, which is fine at < 10 Hz preview rates but not at hashpipe burst rates. |
| **Redis Pub/Sub** | Redis already deployed; decouples producer and consumers. | Pub/Sub is fire-and-forget; lost messages during any disconnect; no cursor semantics; no per-consumer rate shaping. Worse than today. |
| **Redis Streams (`XADD MAXLEN ~ N`)** | Gives bounded ring buffer + consumer groups + cursor. Operationally visible. Uses existing Redis. | Producer side has to serialize `PanoImage` to bytes; consumers need a new client library. Still solving a problem we don't have yet. |
| **RabbitMQ / NATS / Kafka** | Enterprise fan-out, persistence, routing. | New service to operate. Justified only if we hit high-fanout (>100 subscribers) or need durable replay, neither of which is true. |

**Evolution path, if demand materializes (>10 consumers, or durable replay needed):**
- Move to **Redis Streams** (not pub/sub) inside the existing Redis. Producer: `XADD movie:{module_id} MAXLEN ~ 1000 * PanoImage`. Consumer: `XREADGROUP` with its own ID, preserving the cursor semantic.
- Keep `StreamImages` as the gRPC façade; its backend swaps from `latest_data_cache` to `XREADGROUP`. Consumer code does not change.

**What to do now (cheap wins):**
- Add Prometheus gauges to `HpIoManager`: `frames_produced`, `frames_dropped_upstream` (Queue full at `hp_io_manager.py:48`), per-reader `frames_sent` and `last_sent_id_lag` so we can see fanout pressure before we need to act on it.
- Document the lossy-overwrite contract in `grpc/docs/daq_data_service.md`. Today this is implicit.

### 2.2 Telemetry — split the two paths, ship logs via Grafana Alloy

Today's Telemetry service actually runs **two** unrelated pipelines through one gRPC service:

| Path | Flow | Purpose |
|---|---|---|
| `Log` | app → `AsyncGrpcHandler` → gRPC → `RedisBatcher` → Redis LIST `logs:ingress` → `storeLoki.py` → gzip → POST Loki | Structured operational logs |
| `ReportStatus` | app → gRPC → Pydantic `TelemetryConfig` validation → Redis HASH (`{DEVICE_TYPE}_{id}`) → `storeInfluxDB.py` → InfluxDB → Grafana | Device status / housekeeping snapshots |

These deserve different treatment.

#### What does "most elegant" mean here?

The stack we **must** keep: **Redis** (telemetry HASH feeding InfluxDB), **InfluxDB** (metrics backing Grafana dashboards), **Grafana** (operator UI), **Loki** (already deployed at `grafana/loki/docker-compose.loki.yml` and bundled into the telemetry compose). We are not adding any of these — they're here.

The cleanup removes more than it adds:

| Component | Before | After | Delta |
|---|---|---|---|
| `telemetry.Log` proto RPC | present | deleted | -1 RPC |
| `AsyncGrpcHandler` in `telemetry/logger.py` | 77 LOC | deleted | -77 LOC |
| `RedisBatcher` in `telemetry/server.py:45-110` | 65 LOC | deleted | -65 LOC |
| `storeLoki.py` daemon | 265 LOC | deleted | -265 LOC |
| Redis LIST `logs:ingress` / `logs:processing` | custom reliable queue | deleted | one fewer custom protocol |
| Log shipper binary | none (custom Python) | **Grafana Alloy** (~100 MB RSS, one Go binary per node) | +1 off-the-shelf binary |
| Total custom code removed | — | — | **~407 LOC + one proto RPC + one Redis queue convention** |

**Shipper choice — Grafana Alloy over Promtail or Vector.**

| Candidate | Pros | Cons | Verdict |
|---|---|---|---|
| **Promtail** | Smallest footprint (~50 MB); Loki-native; trivial config | **Promtail entered LTS in Feb 2025 and reaches EOL March 2026.** Grafana Labs has officially told everyone to migrate to Alloy. We are already past EOL as of today (2026-04-23). | **Rejected — EOL.** |
| **Vector** | Multi-backend (Loki, Kafka, S3, …); powerful VRL transforms | Operated by Datadog; not Grafana-native; larger cognitive load for one-destination case | Possible fallback, not preferred. |
| **Grafana Alloy** | Unified successor to Promtail + Grafana Agent + OTel Collector; single binary; ships to Loki **and** Prometheus **and** InfluxDB **and** OTLP; actively maintained; config is declarative (river / HCL-style) | ~2× the RSS of Promtail; newer so fewer blog posts | **Recommended.** |

Why Alloy specifically, not Vector:
- **Future consolidation path.** Today `storeInfluxDB.py` is a custom Python daemon that reads Redis hashes and pushes to InfluxDB. Alloy has a native InfluxDB exporter and a native Redis scraper via `prometheus.exporter.redis` and `otelcol.exporter.influxdb` components. A later pass could retire `storeInfluxDB.py` the same way this pass retires `storeLoki.py` — **without introducing a third shipper**.
- **We already run Grafana.** Alloy is the same vendor's collector; Grafana dashboards include first-class Alloy health panels out-of-the-box.
- **OTel compatibility.** If PanoSETI ever wants distributed traces (e.g., tracing a `pseti start` across head + DAQ nodes), Alloy speaks OTLP natively. Promtail does not.

Do **not** go to Vector unless Alloy's river-style config becomes a blocker — Vector is fine, but it gives us nothing over Alloy for PanoSETI's stack and it's a separate vendor from Grafana.

#### How the unified log aggregator works after deletion

Every service continues to call `get_logger("service_name")` (unchanged API). `PanosetiLogFactory` is reconfigured:

- `RichHandler` (console) — unchanged
- `RotatingFileHandler` (local disk) — **promoted to primary shipping source**, emits structured JSONL to `$PANOSETI_LOG_DIR/{service}.jsonl` with one line per log event (timestamp, level, service, message, git_commit, hostname, pid, thread, plus any structured fields passed via `logger.info(msg, extra=...)`)
- `AsyncGrpcHandler` — **removed**

Each node runs Alloy (systemd unit on DAQ nodes, container on the head node — both compose files already run the Grafana stack, we add one service):

```river
// /etc/alloy/config.alloy (abridged)
local.file_match "panoseti" {
  path_targets = [{ "__path__" = "/var/log/panoseti/*.jsonl", "job" = "panoseti" }]
}

loki.source.file "panoseti" {
  targets    = local.file_match.panoseti.targets
  forward_to = [loki.process.panoseti.receiver]
}

loki.process "panoseti" {
  stage.json {
    expressions = { service = "service", level = "level", git_commit = "git_commit", run_id = "run_id" }
  }
  stage.labels {
    values = { service = "", level = "", git_commit = "" }
  }
  stage.static_labels {
    values = { hostname = env("HOSTNAME") }
  }
  stage.output { source = "message" }
  forward_to = [loki.write.headnode.receiver]
}

loki.write "headnode" {
  endpoint { url = "http://headnode:3100/loki/api/v1/push" }
  external_labels = { cluster = "panoseti" }
}
```

Grafana queries are unchanged — Loki's LogQL over labels `{service="daq_control", hostname="daqnode-1"} |= "error"` returns the same results as today, usually in less time (fewer hops).

#### Latency comparison

Tail-to-queryable latency, end-to-end:

| Path | P50 | P99 | Notes |
|---|---|---|---|
| Current custom (gRPC→Redis→storeLoki→Loki) | ~1.0 s | ~2.5 s | `RedisBatcher` holds up to 500 ms (`BATCH_INTERVAL = 0.5`); `storeLoki.py` batches 10 lines / 1 s / 512 KB (`storeLoki.py:85-141`); plus Loki ingest latency |
| **Alloy** (`loki.source.file` → `loki.write`) | **~0.4–1.2 s** | **~1.8 s** | Default WAL flush is 1 s; `max_wait` tunable down to ~250 ms; Loki ingest is the same. Embeds the same Loki write path Promtail used |
| Promtail (EOL) | ~0.4–1.0 s | ~1.5 s | Reference only — not using |
| Vector (hypothetical alt) | ~0.4–1.0 s | ~1.5 s | Similar ballpark; not chosen |

**Alloy matches or beats the current custom path on latency** while removing ~400 LOC. Observatory logs aren't a real-time surface — tail-latency in the low seconds is fine.

#### Do we still need Loki?

**Yes, keep Loki.** The alternatives are worse:

| Alternative | Viable? | Why not |
|---|---|---|
| Put logs in InfluxDB | No | InfluxDB is a timeseries metrics store; log-line cardinality (unique messages, stack traces) explodes the index. Queries like "show me errors in the last hour" are slow. Not designed for free-text search. |
| Elasticsearch / OpenSearch | Overkill | ~2 GB RSS minimum, JVM, complex ops. We don't need full-text-analytic search, just time-ranged tail. |
| ClickHouse | Overkill | Columnar analytical DB; excellent but new service to operate; no existing Grafana provisioning. |
| SSH + `grep` on each DAQ node | Regressive | What we had pre-Loki. Does not scale beyond 2 nodes. No time-range, no filters, no dashboards. |
| Keep Loki (already deployed) | **Yes** | Purpose-built for logs, integrated with Grafana, bundled in our compose, supports LogQL which is strictly richer than what we query today |

Loki is already in the stack — this proposal does **not** add it. The gRPC Log path is the addition that predates Loki's maturity and is now redundant with the standard shipper + Loki combination.

#### ReportStatus path — **keep the custom gRPC service**

- Structured device telemetry with strict Pydantic schemas per device type. Alloy / Vector can't enforce this.
- Hybrid tiered storage (Redis HASH hot + InfluxDB cold) is domain-specific routing.
- `DEV_` prefix / sandbox namespace (`server.py:193-306`) is PanoSETI-specific.
- Traffic is low (one report per device per few seconds). No operational pressure.

What we do clean up: move `TelemetryClient` off `threading.Thread` + `queue.Queue` onto `grpc.aio` + bounded `asyncio.Queue` once the Log path is gone. (Today the thread exists partly because the handler has to be usable from sync code; once logs go via Alloy, sync callers no longer need the gRPC client.)

#### Migration outline (Log path only)

1. Promote `RotatingFileHandler` to the primary shipping source; ensure JSONL formatter emits all fields today carried over gRPC (`service`, `git_commit`, `hostname`, `pid`, `thread`).
2. Deploy Alloy on each node (systemd on DAQ nodes, container on head node), add to existing Loki compose.
3. **Shadow period (2 weeks / 2 observing cycles):** run both paths in parallel. Instrument both with a counter of log lines emitted / queryable in Loki. Any divergence > 0.1% blocks cutover.
4. Cutover: flip `telemetry.log_backend` config from `"grpc"` to `"promtail"`; stop `storeLoki.py`; delete `telemetry.Log` RPC, `AsyncGrpcHandler`, `RedisBatcher`.
5. Keep the local `RichHandler` (console) and `RotatingFileHandler` (disk) paths intact — operator UX is unchanged; `less /var/log/panoseti/daq_control.jsonl` still works.

---

## Phase 3: Transactional Integrity & Chaos/HW Test Plan

### 3.1 Evaluation of the current Start/Stop + transfer design

**Strong:**
- WAL-pattern ledger flips (`ABORTED` written *before* rollback runs, `start.py:120-125`).
- Two-lock hierarchy correctly separates control-plane latency from bulk-I/O duration.
- All transfer-queue transitions use `os.rename` (POSIX-atomic).
- `TransferQueue.enqueue` is idempotent across all four subdirs (`queue.py:114-121`).
- Stale-PID self-healing of `panoseti_control.lock` (`run_state.py:55-96`, SC-015/SC-021 referenced).

**Weak:**
- **VERIFYING is a stub.** `utils/transfer/daemon.py:153` trusts rsync exit code only. `utils/transfer/verify.py::verify_manifest` exists, supports 3-col/4-col manifests and xxh3-128 / SHA-256 / blake3, but no caller wires it in. **This is the single biggest integrity gap** — we can `CleanupData(mode=CLEANUP_SELECTIVE)` off a corrupt transfer.
- No per-attempt exponential backoff in `transfer_daemon.py`; `POLL_INTERVAL_SEC=5.0` is fixed.
- `STOPPED_WITH_ERRORS` has no operator-facing alert path beyond the ledger TOML.
- `run_state.py` advertises `StartTransaction`/`StopTransaction` in docstrings but the actual classes live in `start.py:77`/`stop.py:62`. TRANSACTIONS.md is stale on this.

### 3.2 Manifest-based transfer: the "no deletion without verified integrity" invariant

Wire `verify.py` into the daemon and tighten the RPC contract so the DAQ node *itself* refuses to delete data without proof.

**Flow (the bold steps are the new additions / fixes):**

1. `pseti stop` enqueues `{run_name}.job.toml` to `tmp/transfer_queue/pending/` (already implemented).
2. Daemon claims → `active/` (already implemented).
3. **Stage 1 MANIFEST_GENERATING** (already implemented): parallel `GenerateManifest(algorithm="blake3")` per module via TaskGroup. Manifest written atomically to `{run_dir}/manifest.blake3` on each DAQ node. Record `manifest_path`, `manifest_bytes`, and the **root digest** (blake3-of-manifest-file) into `NodeReceipt`.
4. **Stage 2 TRANSFERRING** (already implemented): rsync each node's run dir to head node, *including* the manifest file. After rsync, re-hash the local copy of the manifest file; compare to the root digest recorded pre-rsync. **Mismatch → `VERIFY_FAILED`, no retry (manifest is small; a corrupt transfer indicates something nasty).**
5. **Stage 3 VERIFYING [NEW — wire `verify.py`]**: for every entry in the local manifest, recompute blake3 of the local file, compare `(digest, size, mtime_ns)`. Any mismatch → `VERIFY_FAILED` → surface to `STOPPED_WITH_ERRORS`. Skip cleanup. Preserve DAQ-side data for manual recovery.
6. **Stage 4 CLEANING [HARDEN RPC]**: extend `CleanupData` to require `manifest_digest: bytes` when `mode=CLEANUP_SELECTIVE`. DAQ server re-reads its local `manifest.blake3`, recomputes the root digest, and *refuses* the RPC with `FAILED_PRECONDITION` unless the digests match. This closes the loop — the head node has to prove it verified the same manifest the DAQ node generated, or deletion is impossible.
7. **Stage 5 ARCHIVED** (already implemented): `run_complete` marker written; ledger → ARCHIVED. **Add** the manifest root digest to `run_complete` for forensic provenance.

**Retry ladder:**
- Stage 2 rsync failure → `TRANSFER_FAILED`, retry up to `MAX_ATTEMPTS=3` with exponential backoff (**new**: 5 s, 30 s, 2 m).
- Stage 3/4 failure → `VERIFY_FAILED` / `STOPPED_WITH_ERRORS` → **no retry**, flag for human.

### 3.3 Chaos test suite — 7 new scenarios in `ci/integration/scenarios/`

All use the existing `chaos/` harness (`grpc_proxy`, `process_chaos`, `netem`, `iptables`, `clock_chaos`, `disk_chaos`) — no new fault-injection plumbing required. IDs continue the existing `SC` scheme; new block `SC095–SC108` under `test_sc_transfer_daemon.py` (new file).

| ID | Name | Target | Fault injection | Pass criteria |
|---|---|---|---|---|
| SC-TX-001 | Partial-start rollback (3 of 10 nodes fail) | `StartTransaction` rollback ladder | `grpc_proxy.failure_mode=UNAVAILABLE` on 3 random nodes during `StartDaq` | Remaining 7 nodes receive `StopDaq`; Quabo flow halted; local daemons killed; ledger=`ABORTED`; `_aborted/{run_name}_0/failure_context.json` present; no orphan PFF on any node |
| SC-TX-002 | Head-node crash mid-start | Ledger / lock recovery | SIGKILL orchestrator after `STARTING` written but before last `StartDaq` returns | Next `pseti start` heals stale lock (`run_state.py:55-96`), archives the abandoned run into `_aborted/`, brings up cleanly |
| SC-TX-003 | Network drop mid-rsync | Transfer retry ladder | `netem.py` 100% packet loss on `daqnode_net` during Stage 2 | Rsync fails → `TRANSFER_FAILED` → backoff → partition lifted → attempt 2 succeeds → `ARCHIVED`; after MAX_ATTEMPTS exhausted, `failed/` holds job, DAQ-side PFF intact |
| SC-TX-004 | Manifest mismatch detected | VERIFYING stage (§3.2) | `docker exec` mutates one byte of a PFF on the DAQ node *after* `GenerateManifest` and *before* rsync | Stage 3 detects digest mismatch → `VERIFY_FAILED` → `STOPPED_WITH_ERRORS`; *no* `CleanupData` executed; DAQ-side data preserved |
| SC-TX-005 | Daemon crash mid-transfer | Resume / `active/` recovery | SIGKILL transfer daemon during Stage 2; restart | Daemon re-claims job from `active/` (this will *find the gap* if not implemented — today `claim()` moves only from `pending/`), retries rsync, succeeds |
| SC-TX-006 | Concurrent `pseti stop` invocations | Control-lock contention | Fire two `pseti stop` in parallel | One wins, the other errors out cleanly with the contender's PID; exactly one job enqueued; no double teardown |
| SC-TX-007 | Cleanup refused without verified manifest | `CleanupData` precondition (§3.2 step 6) | Call `CleanupData(mode=CLEANUP_SELECTIVE)` directly via gRPC with a wrong `manifest_digest` | Server responds `FAILED_PRECONDITION`; PFF files untouched on DAQ |

Each scenario follows the existing pattern: `conftest.py` fixtures spin up the Docker fleet, `chaos/*` utilities inject the fault, `state_probe.py` asserts ledger + queue + filesystem. Target budget: ~200 LOC per scenario.

### 3.4 Hardware-Software test suite — 5 new tests in `ci/hardware-software/`

Target topology (per `TEST-HW-SW.md`): 1 Beelink headnode + 1 DAQ node + 4 Quabos + White-Rabbit switch. Existing `hw_safety_net` session fixture (`conftest.py` runs `pseti validate`, then `pseti stop --force-cleanup` + `pseti power off` at teardown) is the invariant guard. Add these in `test_02_*` through `test_06_*`.

| ID | Name | What it exercises | Assertions |
|---|---|---|---|
| HW-01 | Full start→stop→archive on real Quabos | End-to-end integration (Start/Stop transactions + transfer daemon + selective cleanup) against real hardware | `pseti session-start` → `pseti start --run-type=test --duration=30s` → wait → `pseti stop` → poll ledger until `ARCHIVED` → assert PFF on head node, `run_complete` present, `manifest.blake3` digests match on both sides, DAQ-side `*.pff` gone, DAQ-side `*.json`/`*.log` preserved |
| HW-02 | Quabo power-cycle recovery | WPS power path, UID cache invalidation, HV re-init, MAROC calibration | `pseti power off --module=M` → ping sweep confirms Quabos offline → `pseti power on --module=M` → wait `HW_TEST_QUABO_BOOT_WAIT` → ping sweep confirms all 4 up → `pseti start --duration=10s` succeeds with fresh UIDs |
| HW-03 | White-Rabbit timing sanity | WR firmware + combined-timing path in `utils/pff.py` under real clocks | Run 60 s acquisition → sample N PFF frames → extract `pkt_nsec`, `tv_usec`, `tv_sec` → assert `abs(pkt_nsec/1e6 - tv_usec/1000) < 25 ms` for ≥99% of sampled frames; ±1 s `tv_sec` adjustment applied correctly per the §Timing rule in CLAUDE.md |
| HW-04 | Mid-run hashpipe crash → rollback on real hardware | `StopTransaction` resilient teardown + stale-PID handling against physical state | Start a run → SIGKILL `hashpipe` on DAQ node via SSH mid-run → `pseti status` detects crash → `pseti stop --force-cleanup` → assert Quabos stopped emitting (listen on UDP for 5 s, zero packets), HV ramped down (query WPS), local daemons gone, ledger ∈ {`ABORTED`, `STOPPED_WITH_ERRORS`} with actionable `failure_context.json` |
| HW-05 | Manifest round-trip + corruption detection on real volumes | §3.2 VERIFYING stage against real multi-GB datasets | After HW-01 completes: read `manifest.blake3` on DAQ + head, verify root digests match; manually flip one byte in one PFF on head node; invoke `verify_manifest` directly; assert mismatch detected, exact file surfaced in error |

Gate: these register in `qa.toml` under `suites.test-hw`. `pseti test hw run` is the entry point. No pytest marker; opt-in via CLI, consistent with the existing convention.

---

## Phase 4: Test Migration Plan

These changes touch ~700 existing tests across `control/ci/unit/` (524), `control/ci/integration/` (65), `control/ci/integration/scenarios/` (114), `grpc/tests/` (~200 based on layout). We cannot refactor blindly or we lose the regression net; we cannot keep all of them or we carry dead weight. The answer is a **triage-shadow-cutover** pipeline.

### 4.1 Triage framework — every test is one of three labels

| Label | Definition | Example |
|---|---|---|
| **UPDATE** | Test's *intent* still valid; only the *interface it calls* changes. Preserve the assertion; swap the call. | `test_daq_data_ping` asserts liveness — swap `client.ping()` for `HealthClient.check("panoseti.daq_data")`. Assertion unchanged. |
| **REFACTOR** | Test's *intent* still valid but the *system under test* is being replaced. Rewrite against the new implementation. | `test_loki_pipeline.py` today asserts gRPC→Redis→Loki round-trip. Rewrite to assert file-write→Alloy→Loki round-trip. |
| **DELETE** | Test validates an *implementation detail that ceases to exist*. Coverage is either meaningless (the code is gone) or redundant with a remaining test. | Unit tests for `RedisBatcher`, `AsyncGrpcHandler.emit`, `storeLoki.py::BLMOVE`. |

### 4.2 Triage classifier — automated first pass

Before any human reads a test, run a small classifier script (`ci/scripts/classify_tests.py`, ~150 LOC) that labels every test function by parsing imports and function bodies:

```
for each test file:
    if imports any of {RedisBatcher, AsyncGrpcHandler, storeLoki, telemetry.Log stub, daq_data.Ping stub, grpc channel_ready probe via ublox}:
        if the import is the primary subject → DELETE
        else → UPDATE (it's a peripheral dep)
    elif imports grpc_utils replacement surface we haven't written yet:
        → already migrated, skip
    elif calls `asyncio.gather(` in its own body (not under test):
        → UPDATE (the pattern changes in prod code but the test assertion likely stays)
    elif test file name matches test_loki* | test_telemetry_log* | test_redis_batcher* | test_aws_grpc_handler*:
        → REFACTOR
    else:
        → KEEP (no change expected)
```

Output: one CSV of `file, test_name, current_label, reviewer_label, status` committed to the repo. Reviewer can override the label; CI gates on this file.

### 4.3 Expected distribution (estimated from exploration)

| Suite | Total | UPDATE | REFACTOR | DELETE | KEEP |
|---|---|---|---|---|---|
| `control/ci/unit/` | ~524 | ~60 (gRPC error types, TaskGroup outcome handling, `client_models` propagation) | ~8 (transfer daemon VERIFYING wiring) | ~8 (old retry loop, SSH cleanup paths if any touch deprecated code) | ~448 |
| `control/ci/integration/` | ~65 | ~15 (Ping → Health, error unwrapping) | ~3 (`test_loki_pipeline.py`, `test_daq_data_ping.py` if exists) | ~4 | ~43 |
| `control/ci/integration/scenarios/` | ~114 | ~8 (SC056–SC068 telemetry scenarios now target Alloy outage modes) | ~5 | ~0 | ~101 |
| `grpc/tests/telemetry/` | ~50 | ~10 | ~10 (rewrite to Alloy E2E) | ~20 (RedisBatcher, AsyncGrpcHandler, Log-stub pytest fixtures) | ~10 |
| `grpc/tests/daq_data/` | ~60 | ~15 (Ping → Health; `__aexit__` cancellation tests) | ~5 | ~2 | ~38 |
| `grpc/tests/daq_control/` | ~70 | ~25 (client_models propagation, typed exceptions, CleanupData manifest_digest precondition) | ~3 | ~2 | ~40 |
| `grpc/tests/ublox_control/` | ~20 | 0 (excluded) | 0 | 0 | ~20 |
| **Total** | **~903** | **~133 (~15%)** | **~34 (~4%)** | **~36 (~4%)** | **~700 (~78%)** |

Only ~4% of tests get deleted outright. The rest are either unchanged (76%) or mechanical updates (16%).

### 4.4 Shadow-period strategy — avoid regression blindness

The risk is deleting old tests before new tests cover the same invariants. The fix: **both paths live simultaneously for one soak period.**

```
Weeks 1–2  Implement grpc_utils + Health + manifest verify (new code behind feature flag)
           Classifier run → triage CSV landed in repo
           New tests written against new code (targeting +UPDATE labels)
           Old tests still green against old code

Weeks 3–4  Deploy Alloy in shadow mode (writes to Loki in a parallel namespace)
           storeLoki.py still writes to primary Loki namespace
           Dual-path log diff counter instrumented
           All old + new tests must pass in CI

Weeks 5–6  Cutover: feature flag → new. storeLoki.py disabled. Alloy → primary.
           Shadow soak (no production flip yet): 7 days of full observing cycles.
           Any log-count divergence > 0.1% or any new regression blocks cutover.

Week 7    Remove old code (AsyncGrpcHandler, RedisBatcher, storeLoki.py,
           telemetry.Log RPC, daq_data.Ping RPC).
           Delete DELETE-labeled tests in one PR.
           UPDATE-labeled tests are rewritten in the same PR as the code change.
```

### 4.5 Regression gates

- **Coverage gate:** `pytest --cov` on `utils/` + `grpc_utils/` cannot drop below pre-migration coverage minus 1 %.
- **Count gate:** after each cutover PR, `passing_tests_new >= passing_tests_before - DELETE_count`. CI fails if the ratio drops further.
- **Behavior diff gate:** during shadow period, a nightly job re-runs the last 24 h of observing against both Log paths and compares line counts + unique-message-sets. Must be within 0.1%.
- **Triage-CSV gate:** every deleted test must have an entry in the CSV with `status=deleted` and a one-line rationale. CI rejects deletions of tests not pre-classified.

### 4.6 Concrete per-change impact

| Change | Tests UPDATE | Tests REFACTOR | Tests DELETE | Notes |
|---|---|---|---|---|
| `grpc_utils.exceptions` (typed errors) | ~40 (every `except ConnectionError` assertion → `except PanosetiRpcError`) | 0 | 0 | Mechanical, scriptable with a codemod |
| Sync/async method consolidation in `daq_control/client.py` | ~20 (tests that patched one of the sync copies still work; imports may shift) | 0 | 0 | |
| `HealthServicer` adoption | ~15 (Ping/channel_ready probes → HealthClient) | 0 | 0 | |
| `client_models` propagation across boundaries | ~25 (test fixtures now construct Pydantic models, not raw dicts) | 0 | 0 | Fixture refactor |
| `TaskGroup` migration for `stop.py`/`start.py` `gather` | ~10 (outcome-collection shape) | 0 | 0 | |
| Remove `AsyncGrpcHandler` / `RedisBatcher` / `storeLoki` | 0 | ~12 (log pipeline E2E) | ~30 (internal unit tests of removed classes) | The bulk of DELETEs concentrate here |
| Manifest VERIFYING wiring + `CleanupData` precondition | ~3 (transfer daemon unit tests) | ~3 (transfer E2E) | 0 | Plus 7 new chaos tests in §3.3 and 1 HW test in §3.4 |
| Remove `daq_data.Ping` RPC | ~8 | 0 | ~4 | Deprecation window: one release |
| **Total** | **~121** | **~15** | **~34** | Matches §4.3 estimates within ±10% |

### 4.7 Dead-weight audit (independent of this refactor)

While we're in the test tree, do a separate pass (half-day) to find tests that are already dead weight today — unrelated to this plan:

- Duplicate scenarios (classifier emits a `DUPLICATE` candidate list based on assertion fingerprinting).
- `@pytest.mark.skip` tests older than 6 months without an issue link — either revive or delete.
- Tests that only assert log-line formatting (fragile, low value — migrate to assertion-free smoke if the signal still matters).

Budget: ~20 more deletions expected. Commit separately so the migration-attributed deletions stay measurable.

---

## Critical files to modify

New code:
- `grpc/src/panoseti_grpc/grpc_utils/{channel,exceptions,decorators,interceptors,retries,health}.py`
- `control/src/control/ci/integration/scenarios/test_sc_transfer_daemon.py` (SC-TX-001..007)
- `control/src/control/ci/hardware-software/test_02_*..test_06_*.py` (HW-01..05)

Modified code:
- `grpc/src/panoseti_grpc/{daq_data,daq_control,telemetry}/client.py` — thin down to service-specific methods using `grpc_utils` (`ublox_control` excluded)
- `grpc/src/panoseti_grpc/server.py` — register `HealthServicer`
- `grpc/protos/daq_control.proto` — add `manifest_digest` to `CleanupDataRequest`
- `grpc/src/panoseti_grpc/daq_control/server.py::CleanupData` — enforce precondition
- `control/src/control/utils/transfer/daemon.py` — wire `verify.py::verify_manifest` into Stage 3; add exponential backoff; resume `active/` jobs on restart
- `control/src/control/stop.py:361`, `control/src/control/start.py:725` — replace `asyncio.gather` with outcome-collection under `TaskGroup`
- `grpc/src/panoseti_grpc/daq_data/client.py:635-644` — stop swallowing `CancelledError`
- `control/TRANSACTIONS.md` — correct `StartTransaction`/`StopTransaction` location

Files to remove (after soak, Phase 2.2):
- `grpc/src/panoseti_grpc/telemetry/logger.py::AsyncGrpcHandler`
- `grpc/src/panoseti_grpc/telemetry/server.py::RedisBatcher`
- `control/src/control/daemons/storeLoki.py`
- `telemetry.Log` RPC from `protos/telemetry.proto`

---

## Verification

Local (no hardware, no Docker):
```bash
cd control && uv run pytest ci/unit/ -v                            # 524 unit tests
cd ../grpc && python tests/qa.py all                               # all lint + service tests
```

Docker CI (chaos):
```bash
cd control && pseti test sw chaos -k "SC_TX"                       # run the 7 new scenarios
pseti test sw integration -k "not SC_TX"                           # regression on 65 passing
```

Hardware (opt-in, real Quabos):
```bash
cd control && pseti test hw check-env                              # verify Beelink + DAQ + Quabos reachable
pseti test hw run -k "HW_01"                                       # smoke first
pseti test hw run                                                  # full HW-01..05
```

Manual validation of Phase-2.2 Log migration (after Alloy deploy):
```bash
# 1. Write a probe log via get_logger on each node
# 2. Confirm JSONL line appears in $PANOSETI_LOG_DIR/{service}.jsonl
# 3. Confirm same line queryable in Loki:
#      logcli query '{service="daq_control", hostname="daqnode-1"}' --since=1m
# 4. Kill Loki; write more logs; confirm Alloy buffers locally (positions.yaml + WAL);
#    restart Loki; confirm buffered lines drain within the batchwait window.
# 5. Shadow-period diff check:
#      python ci/scripts/log_diff.py --since=24h  # counts + unique-message-set diff
#    Must report < 0.1% divergence before cutover is unblocked.
```

Test-migration verification:
```bash
python ci/scripts/classify_tests.py --write-csv ci/test_triage.csv  # triage classifier
pytest --collect-only | wc -l                                        # enforce count gate
pytest --cov=utils --cov=grpc_utils --cov-report=term-missing        # enforce coverage gate
```

Success criteria:
- All (524 + 7 new chaos + ~121 updated) unit, (65 + 7) integration, and 5 HW tests pass.
- `grpc_utils` package replaces ≥ 500 LOC of duplicated boilerplate across the four services (measurable via `tokei` pre/post).
- `grpc_health_probe -addr=<node>:50051 -service=panoseti.daq_control` succeeds on every node, replacing custom `Ping` / `channel_ready` probes.
- `CleanupData` with a missing or wrong `manifest_digest` is rejected 100% of the time (HW-05 and SC-TX-007 confirm).
- Alloy soak: log-line divergence vs. parallel gRPC path < 0.1% across a 7-day observing window; P99 end-to-end latency < 2 s.
- Test triage CSV has a labeled entry for every deleted test; `passing_tests_new >= passing_tests_before - DELETE_count` holds across every cutover PR.

---

## Sequencing — suggested PR order

Doing this in one PR is suicide. Doing it in the wrong order causes cascade failures. Suggested order (each is independently mergeable):

1. **`grpc_utils/exceptions.py` + `@rpc_method` decorator** — landing typed exceptions first unlocks mechanical codemods for all the `except ConnectionError` sites. Test impact: ~40 UPDATE.
2. **`grpc_utils/channel.py` + `MultiHostChannelManager`** — extract channel lifecycle; migrate `daq_data` then orchestrator. Test impact: ~20 UPDATE.
3. **`grpc_utils/health.py` + `HealthServicer` adoption** — add behind flag; keep `Ping` until consumers migrate. Test impact: ~15 UPDATE.
4. **`daq_control/client.py` sync/async consolidation + `client_models` propagation** — landlocks Pydantic models across the boundary. Test impact: ~45 UPDATE.
5. **Transaction fixes: `TaskGroup` migration + `__aexit__` cancel bug + manifest VERIFYING wiring + `CleanupData` precondition** — the correctness fixes. Ship the 7 chaos tests in the same PR. Test impact: ~13 UPDATE + 7 new.
6. **Alloy shadow deploy** — no code removal yet; dual-path logging active. Test impact: ~10 new (log diff harness).
7. **Alloy cutover** — flip feature flag; monitor.
8. **Delete old log path** — remove `AsyncGrpcHandler`, `RedisBatcher`, `storeLoki.py`, `telemetry.Log` RPC. Delete ~30 tests in the same PR. Test impact: ~12 REFACTOR + ~30 DELETE.
9. **Delete `daq_data.Ping` RPC after one release deprecation window.** Test impact: ~4 DELETE.
   Note: `ublox_control` service is explicitly out of scope for this refactor — its `channel.channel_ready()` probes, `__aexit__` cancel suppression, and test suite are left untouched.
10. **HW tests HW-01..05 + `pseti test hw run` integration**. Test impact: 5 new.

Risk-weighted cost: steps 1–4 are pure refactor (low risk, high LOC churn). Step 5 is the correctness fix (medium risk, needs all 7 chaos tests green before merge). Steps 6–8 carry the operational risk (shadow soak gates this). Steps 9–10 are cleanup.
