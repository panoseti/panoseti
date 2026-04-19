#!/usr/bin/env python3
"""Transfer daemon entry point — launched by session_start.py via util.start_daemon()."""
from __future__ import annotations

import asyncio
import logging

# Ensure control/ is on sys.path so ``utils`` is importable.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from control.utils.transfer.daemon import run_daemon  # noqa: E402

if __name__ == "__main__":
    asyncio.run(run_daemon())
