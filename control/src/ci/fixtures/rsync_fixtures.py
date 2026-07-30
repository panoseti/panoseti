"""
ci/fixtures/rsync_fixtures.py

Fixtures for mocking rsync subprocess calls in transfer tests.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class RsyncMock:
    """Configurable mock for rsync subprocesses."""
    def __init__(self):
        self.return_code = 0
        self.stdout_lines = [b"1000 50% 10MB/s 0:01\n", b""]
        self.stderr_content = b""
        self.delay = 0.0
        self.side_effect = None
        self.call_count = 0

    def __call__(self, *args, **kwargs) -> Any:
        self.call_count += 1
        
        if self.side_effect:
            res = self.side_effect(*args, **kwargs)
            # If side effect returns an awaitable, it must be handled by the caller (AsyncMock)
            # but here we are a side_effect of an AsyncMock.
            if res is not None:
                return res
            
        return self._make_proc_mock()

    def _make_proc_mock(self) -> MagicMock:
        proc = MagicMock()
        proc.returncode = self.return_code
        proc.wait = AsyncMock(return_value=self.return_code)
        proc.communicate = AsyncMock(return_value=(b"".join(self.stdout_lines), self.stderr_content))
        proc.stdout = MagicMock()
        proc.stdout.readline = AsyncMock(side_effect=self.stdout_lines)
        proc.stderr = MagicMock()
        proc.stderr.read = AsyncMock(return_value=self.stderr_content)
        return proc

    def mock_process_ok(self, *args, **kwargs) -> MagicMock:
        return self._make_proc_mock()

    def mock_process_fail(self, *args, message: str = "rsync failed", **kwargs) -> MagicMock:
        proc = self._make_proc_mock()
        proc.returncode = 1
        proc.wait = AsyncMock(return_value=1)
        proc.communicate = AsyncMock(return_value=(b"", message.encode()))
        proc.stderr.read = AsyncMock(return_value=message.encode())
        return proc

@pytest.fixture
def mock_rsync_transfer():
    """Provides a configurable rsync mock and automatically patches it."""
    rsync_mock = RsyncMock()
    # CRITICAL: Use new_callable=AsyncMock so that 'await create_subprocess_exec' works.
    with patch("control.transfer.daemon.asyncio.create_subprocess_exec", new_callable=AsyncMock, side_effect=rsync_mock) as p:
        yield rsync_mock
