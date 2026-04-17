"""
chaos/grpc_proxy.py

In-process gRPC chaos interceptor.

Wraps a DaqControlClient to inject faults per-method:
  - "timeout"       : raise grpc.RpcError with DEADLINE_EXCEEDED
  - "unavailable"   : raise grpc.RpcError with UNAVAILABLE
  - "slow_response" : sleep timeout_s before delegating to real call
  - "success_then_fail": first call succeeds, subsequent raise UNAVAILABLE
  - "reset_stream"  : raise grpc.RpcError with INTERNAL

Usage:
    proxy = GrpcChaosProxy(real_client)
    proxy.set_mode("StartDaq", "unavailable")
    with proxy:
        # client.StartDaq() will raise UNAVAILABLE
        do_test(proxy)
    # proxy.restore() called automatically
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator
from unittest.mock import patch

logger = logging.getLogger(__name__)

try:
    import grpc
    _GRPC_AVAILABLE = True
except ImportError:
    _GRPC_AVAILABLE = False


def _make_rpc_error(code_name: str, message: str = "") -> Exception:
    """Build a grpc.RpcError with the given status code name."""
    if not _GRPC_AVAILABLE:
        return RuntimeError(f"gRPC {code_name}: {message}")

    class _RpcError(grpc.RpcError):
        def code(self) -> Any:
            return getattr(grpc.StatusCode, code_name, grpc.StatusCode.UNKNOWN)
        def details(self) -> str:
            return message

    return _RpcError()


class GrpcChaosProxy:
    """
    Wraps a DaqControlClient to inject per-method faults.

    All injected faults are context-managed — calling restore() or
    using this object as a context manager undoes all patches.
    """

    def __init__(self, target: Any | None = None) -> None:
        self.target = target
        self._modes: dict[str, tuple[str, float]] = {}  # method → (mode, timeout_s)
        self._call_counts: dict[str, int] = {}
        self._patches: list[Any] = []

    def set_mode(
        self,
        node_or_method: str,
        method_or_mode: str,
        mode: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        """Configure a fault mode for a method.

        Two calling conventions:
          set_mode("StartDaq", "unavailable")         # method, mode
          set_mode("daqnode", "StartDaq", "timeout")  # node, method, mode (node ignored in proxy)
        """
        if mode is None:
            # Two-arg form: node_or_method is actually the method name
            method = node_or_method
            actual_mode = method_or_mode
        else:
            # Three-arg form: node_or_method is node (ignored), method_or_mode is method
            method = method_or_mode
            actual_mode = mode
        self._modes[method] = (actual_mode, timeout_s)
        self._call_counts[method] = 0

    def _wrap_method(self, method_name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        """Return a patched version of method_name that injects the configured fault."""

        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            if method_name not in self._modes:
                return original(*args, **kwargs)

            fault_mode, timeout_s = self._modes[method_name]
            count = self._call_counts.get(method_name, 0)
            self._call_counts[method_name] = count + 1

            if fault_mode == "timeout" or (fault_mode == "slow_response"):
                time.sleep(timeout_s)
                if fault_mode == "timeout":
                    raise _make_rpc_error("DEADLINE_EXCEEDED", f"Injected timeout on {method_name}")
                return original(*args, **kwargs)

            elif fault_mode == "unavailable":
                raise _make_rpc_error("UNAVAILABLE", f"Injected UNAVAILABLE on {method_name}")

            elif fault_mode == "success_then_fail":
                if count == 0:
                    return original(*args, **kwargs)
                raise _make_rpc_error("UNAVAILABLE", f"Injected failure after first success on {method_name}")

            elif fault_mode == "reset_stream":
                raise _make_rpc_error("INTERNAL", f"Injected RST_STREAM on {method_name}")

            elif fault_mode == "partial_response":
                raise _make_rpc_error("DATA_LOSS", f"Injected partial response on {method_name}")

            return original(*args, **kwargs)

        return _wrapper

    def apply(self, client: Any) -> None:
        """Monkey-patch all configured method faults onto client."""
        for method_name in self._modes:
            if hasattr(client, method_name):
                original = getattr(client, method_name)
                patched = self._wrap_method(method_name, original)
                self._patches.append((client, method_name, original))
                setattr(client, method_name, patched)

    def restore(self) -> None:
        """Undo all monkey-patches."""
        for obj, attr, original in self._patches:
            setattr(obj, attr, original)
        self._patches.clear()

    def __enter__(self) -> "GrpcChaosProxy":
        if self.target is not None:
            self.apply(self.target)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.restore()


@contextmanager
def inject_rpc_fault(
    client: Any,
    method: str,
    mode: str,
    timeout_s: float = 30.0,
) -> Generator[None, None, None]:
    """Convenience context manager for single-method fault injection."""
    proxy = GrpcChaosProxy(client)
    proxy.set_mode(method, mode, timeout_s=timeout_s)
    with proxy:
        yield
