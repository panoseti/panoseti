#!/usr/bin/env python3
"""Transfer daemon entry point — launched by session_start.py via util.start_daemon()."""
from __future__ import annotations

import asyncio
import logging
import os
import sys

# Ensure control/ is on sys.path so ``utils`` is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from utils.transfer.daemon import run_daemon

if __name__ == "__main__":
    asyncio.run(run_daemon())
