"""
Trimmed stub of the old software's tools/interleave.py.

The full interleaving-mode module isn't part of this legacy fallback
toolkit's scope (DAQ-stop + quabo data-flow-stop only) -- stop.py imports
just this constant to check for and clean up a stale interleave PID file
during teardown.
"""

PID_FILE = "tmp/interleave.pid"
