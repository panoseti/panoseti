"""Tier 1 (Unit): Logger propagation & idempotency.

Verifies:
- get_logger(name) returns a logger with propagate=False.
- get_logger(name) is idempotent (doesn't add duplicate handlers).
- RichHandler has the correct service tag formatter.
"""
from __future__ import annotations

import logging

from panoseti_grpc.telemetry.logger import get_logger
from rich.logging import RichHandler


def test_logger_propagation_is_disabled() -> None:
    name = "TEST.Propagation"
    logger = get_logger(name)
    assert logger.propagate is False

def test_logger_idempotency() -> None:
    name = "TEST.Idempotency"
    logger1 = get_logger(name)
    handler_count = len(logger1.handlers)
    
    logger2 = get_logger(name)
    assert len(logger2.handlers) == handler_count
    assert logger1 is logger2

def test_logger_formatter_has_service_tag() -> None:
    name = "TEST.Tag"
    logger = get_logger(name)
    rich_handler = next(h for h in logger.handlers if isinstance(h, RichHandler))
    assert rich_handler.formatter is not None
    # Formatting a record to check the output
    record = logging.LogRecord(name, logging.INFO, "path", 10, "hello", None, None)
    formatted = rich_handler.formatter.format(record)
    assert f"[{name}] hello" in formatted

def test_no_root_handler_on_import() -> None:
    # This check ensures that importing start/stop doesn't install a root handler
    # via transitive import of interleave.py
    
    root_logger = logging.getLogger()
    assert not any(isinstance(h, RichHandler) for h in root_logger.handlers)
