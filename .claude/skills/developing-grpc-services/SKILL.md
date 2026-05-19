---
name: developing-grpc-services
description: Use when modifying or adding code under grpc/ (panoseti_grpc) — gRPC servicers, clients, .proto files, the unified server, the daq_data gateway/edge path, or a new service.
---

# Developing gRPC Services

## Overview

All active services (`daq_data`, `daq_control`, `telemetry`) co-host on a single port via `PanosetiServer`. gRPC routes by proto package name — no collision. Source root: `grpc/src/panoseti_grpc/`.

## Deployment profiles

| Profile | Services | Command |
|---------|----------|---------|
| `default` | telemetry + daq_data + daq_control | `pseti-grpc server` |
| `daq_node` | daq_data + daq_control | `pseti-grpc server --profile daq_node` |
| `headnode` | telemetry | `pseti-grpc server --profile headnode` |

## DAQ Data gateway/edge topology

```
Consumer → AioDaqDataClient(headnode, port)
              │
     DaqDataGatewayServicer (headnode)
       ├── AioDaqDataClient(daq-node-1, port)
       └── AioDaqDataClient(daq-node-N, port)
                    │  UDS
           DaqDataServicer ← Hashpipe
```

Consumers always connect to the headnode gateway. Use `AioDaqDataClient(host, port)` — single-target; the gateway handles fan-in.

## Shared machinery

**`grpc_utils/`** — client-side and channel utilities:

| Tool | Module | Purpose |
|------|--------|---------|
| `@grpc_call` | `grpc_utils.decorators` | Wraps client methods; maps `grpc.RpcError → PanosetiRpcError`; never suppresses `CancelledError` |
| `HealthClient` | `grpc_utils.health` | `grpc.health.v1` checks; replaces deprecated `Ping` RPC |
| `PanosetiRpcError` subclasses | `grpc_utils.exceptions` | `UnavailableError`, `DeadlineExceededError`, `FailedPreconditionError`, … |
| `AsyncChannelManager` | `grpc_utils.channel` | Channel lifecycle with keepalive |
| `build_retry_service_config()` | `grpc_utils.retries` | Declarative retry policy |

**`util/error_handling.py`** — server-side handler decorator (NOT in `grpc_utils/`):
```python
from panoseti_grpc.util.error_handling import grpc_error_handler
# Wraps every server RPC handler; catches unhandled exceptions → INTERNAL status
```

## Proto → code workflow

```bash
# Edit protos/ then:
python scripts/compile_protos.py
# Generates generated/*_pb2.py and *_pb2_grpc.py — NEVER edit these by hand
```

## Adding a new service (5-step checklist)

1. Define `.proto`; run compile script.
2. Implement servicer (`server.py`) + client (`client.py`) in a new `src/panoseti_grpc/<name>/` dir.
3. Write `async def _make_<name>_servicer(cfg, shutdown_event)` in `server.py`.
4. Add `<name>: NewServiceConfig` to `PanosetiServerConfig` and `<name>: bool` to `ServiceToggles`.
5. Call `ServiceRegistry.register(ServiceDescriptor(...))` at module level in `server.py`.

No changes to `PanosetiServer` itself needed.

## Key gotchas

See `grpc/CLAUDE.md` Key Gotchas for: `init_sim()` vs `init_hp_io()` ordering, UDS simulation ordering, non-breaking space in README edits, `grpc_error_handler` async-generator detection, `OSError [Errno 5]` overlay2 fix.

## Full references

- `grpc/CLAUDE.md` — architecture, service details, gotchas
- `grpc/GEMINI.md` — engineering standards, concurrency rules
- `grpc/docs/server.md` — unified server config
- `grpc/docs/daq_data_service.md`, `daq_control_service.md`, `telemetry_service.md`
- `grpc/src/panoseti_grpc/grpc_utils/README.md` — concurrency decision framework
