"""Transfer state machine constants and retry policy."""

MAX_ATTEMPTS: int = 3
RETRY_DELAYS: list[int] = [5, 30]  # seconds between attempts 1->2 and 2->3

TRANSFER_STAGES: list[str] = [
    "MANIFEST_GENERATING",
    "TRANSFERRING",
    "VERIFYING",
    "CLEANING",
    "ARCHIVED",
]

ERROR_STAGES: list[str] = [
    "TRANSFER_FAILED",
    "VERIFY_FAILED",
    "STOPPED_WITH_ERRORS",
]
