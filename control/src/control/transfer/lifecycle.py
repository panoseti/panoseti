"""Transfer state machine constants and retry policy."""
from control.utils.pydantic_config_models import RunStatus

MAX_ATTEMPTS: int = 3
RETRY_DELAYS: list[int] = [5, 30]  # seconds between attempts 1->2 and 2->3

TRANSFER_STAGES: list[RunStatus] = [
    RunStatus.MANIFEST_GENERATING,
    RunStatus.TRANSFERRING,
    RunStatus.VERIFYING,
    RunStatus.CLEANING,
    RunStatus.ARCHIVED,
]

ERROR_STAGES: list[RunStatus] = [
    RunStatus.TRANSFER_FAILED,
    RunStatus.VERIFY_FAILED,
    RunStatus.STOPPED_WITH_ERRORS,
]
