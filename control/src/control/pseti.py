from typing import Annotated, Any

import typer
from panoseti_grpc.util.cli import BaseLazyGroup, display_tree_callback


class PanoLazyGroup(BaseLazyGroup):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        lazy_mapping = {
            # Root commands (formerly under 'obs')
            "power": ("control.power", "app", "Control Quabo power via WPS."),
            "uids": ("control.get_uids", "app", "Scan and record Quabo hardware UIDs."),
            "cfg": ("control.config", "app", "Configure observatory hardware and daemons."),
            "val": ("control.config", "validate_app", "Configuration and topology validation tools."),
            "start": ("control.start", "app", "Start a new recording run."),
            "stat": ("control.status", "app", "Show observatory health, acquisition status, and ledger."),
            "stop": ("control.stop", "app", "Stop and finish the current recording run."),
            "xfr": ("control.transfer.cli", "app", "Inspect and manage the background transfer queue."),
            "session-start": ("control.session_start", "app", "Initialize hardware/power for an observing session."),
            "session-stop": ("control.session_stop", "app", "Gracefully power down and terminate a session."),
            # System commands
            "show": ("control.tools.show_cli", "app", "Inspect and visualize system state (paths, commands)."),
            "test": ("ci.test_cli", "app", "Unified PSETI testing suite (lint, sw, hw)."),
            "grpc": ("panoseti_grpc.cli", "app", "gRPC service operations (health, reflection, etc)."),
        }
        # Explicit order to ensure consistent UX regardless of mapping insertion order
        command_order = [
            "power", "uids", "cfg", "val", "start", "stat", "stop", "xfr",
            "session-start", "session-stop", "show", "test", "grpc"
        ]
        super().__init__(
            *args, 
            lazy_mapping=lazy_mapping, 
            command_order=command_order,
            **kwargs
        )


app = typer.Typer(
    cls=PanoLazyGroup,
    help="PSETI Observatory Control CLI",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback()
def main_callback(
    ctx: typer.Context,
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree and exit.", callback=display_tree_callback)] = False
) -> None:
    """PSETI Control Plane."""
    pass

if __name__ == "__main__":
    app()
