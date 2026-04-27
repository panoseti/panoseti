from __future__ import annotations

from typing import Annotated, Any

import typer
from panoseti_grpc.util.cli import BaseLazyGroup, display_tree_callback


class ObsLazyGroup(BaseLazyGroup):
    """
    Lazy-loading group for PSETI Observatory operations.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        lazy_mapping = {
            "start": ("control.start", "app", "Start a new recording run."),
            "stop": ("control.stop", "app", "Stop and finish the current recording run."),
            "status": ("control.status", "app", "Show observatory health and acquisition status."),
            "session-start": ("control.session_start", "app", "Initialize hardware/power for an observing session."),
            "session-stop": ("control.session_stop", "app", "Gracefully power down and terminate a session."),
            "get-uids": ("control.get_uids", "app", "Scan and record Quabo hardware UIDs."),
            "config": ("control.config", "app", "Configure observatory hardware and daemons."),
            "power": ("control.power", "app", "Control Quabo power via WPS."),
            "val": ("control.config", "validate_app", "Configuration and topology validation tools."),
            "transfer": ("control.transfer.cli", "app", "Inspect and manage the background transfer queue."),
            "ledger": ("control.tools.ledger_cli", "app", "Inspect the run state ledger (read-only)."),
            "led": ("control.tools.ledger_cli", "app", "Short alias for 'ledger'."),
        }
        command_order = ["power", "get-uids", "config", "val", "start", "status", "stop", "transfer", "ledger", "led", "session-start", "session-stop"]
        super().__init__(*args, lazy_mapping=lazy_mapping, command_order=command_order, **kwargs)

app = typer.Typer(
    cls=ObsLazyGroup,
    help="Observatory operations (Start/Stop, Power, Config, Validation).",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

@app.callback()
def main(
    ctx: typer.Context,
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree for observatory operations.", callback=display_tree_callback)] = False
) -> None:
    """PSETI Observatory Control."""
    pass
