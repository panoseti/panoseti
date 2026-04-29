import asyncio
import json
import re
import time
from typing import Any

class Result:
    __slots__ = ("code", "elapsed", "name", "stats", "stdout")
    def __init__(self, name: str, code: int, elapsed: float, stats: dict[str, int] | None = None, stdout: str = "") -> None:
        self.name    = name
        self.code    = code
        self.elapsed = elapsed
        self.stats   = stats or {}
        self.stdout  = stdout
    @property
    def ok(self) -> bool: return self.code == 0

async def stream_test_output(
    name: str, 
    proc: asyncio.subprocess.Process, 
    lock: asyncio.Lock, 
    start_time: float, 
    tag: str = ""
) -> Result:
    """
    Streams output from a subprocess, parsing TEST_METRICS_JSON if present,
    and colorizing output using rich.
    """
    assert proc.stdout is not None
    
    stats = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    has_json_metrics = False
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    async for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip()
        plain_line = ansi_escape.sub('', line)
        
        if "TEST_METRICS_JSON: " in plain_line:
            try:
                json_str = plain_line.split("TEST_METRICS_JSON: ")[1]
                summary = json.loads(json_str)
                stats["passed"] = summary.get("passed", 0) + summary.get("xpass", 0)
                stats["failed"] = summary.get("failed", 0)
                stats["skipped"] = summary.get("skipped", 0) + summary.get("xfail", 0)
                stats["error"] = summary.get("error", 0)
                has_json_metrics = True
            except Exception:
                pass

        upper_line = plain_line.upper()
        if not has_json_metrics:
            if " PASSED " in upper_line or " . " in upper_line:
                stats["passed"] += 1
            if " FAILED " in upper_line or " F " in upper_line:
                stats["failed"] += 1
            if " ERROR " in upper_line:
                stats["error"] += 1

        async with lock:
            from rich.console import Console
            # Colorize test statuses for better readability without pytest's native ANSI
            formatted_line = plain_line
            if "PASSED" in formatted_line:
                formatted_line = formatted_line.replace("PASSED", "[green]PASSED[/green]")
            elif "FAILED" in formatted_line:
                formatted_line = formatted_line.replace("FAILED", "[red]FAILED[/red]")
            elif "ERROR" in formatted_line:
                formatted_line = formatted_line.replace("ERROR", "[red]ERROR[/red]")
            
            stream_console = Console(highlight=False, force_terminal=True)
            stream_console.print(f"{tag}{formatted_line}")

    await proc.wait()
    return Result(name, proc.returncode or 0, time.monotonic() - start_time, stats=stats)
