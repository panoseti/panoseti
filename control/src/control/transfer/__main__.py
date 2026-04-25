"""Launch the transfer daemon via `python -m control.transfer`."""
import asyncio
import logging

from control.transfer.daemon import run_daemon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

if __name__ == "__main__":
    asyncio.run(run_daemon())
