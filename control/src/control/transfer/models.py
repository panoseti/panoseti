"""Transfer queue domain models and status enum."""
from __future__ import annotations

from enum import StrEnum

from control.utils.pydantic_config_models import TransferJob, TransferNodeSpec

__all__ = ["TransferJob", "TransferNodeSpec", "TransferStatus"]


class TransferStatus(StrEnum):
    """Lifecycle bucket for a transfer queue job."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"

