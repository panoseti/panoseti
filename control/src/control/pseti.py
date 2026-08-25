import importlib.metadata
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from control.utils.env_loader import load_pseti_env

# Check for --no-env flag early to prevent loading .env files if the user disables it
if "--no-env" not in sys.argv:
    # Load .env variables (if any) before initializing config and commands
    load_pseti_env()

import typer
from panoseti_grpc.util.cli import BaseLazyGroup, display_tree_callback

from control.utils.paths import PanoPaths


def _version_callback(value: bool) -> None:
    if value:
        print(f"pseti {importlib.metadata.version('pseti-ctl')}")
        raise typer.Exit()


def _env_template_callback(value: bool) -> None:
    if not value:
        return
    src = PanoPaths.software_root_dir() / ".env.example"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    dest = Path.cwd() / f".env_pseti_{timestamp}"
    if dest.exists():
        print(f"Refusing to overwrite existing file: {dest}")
        raise typer.Exit(code=1)
    shutil.copyfile(src, dest)
    print(f"Wrote .env template to {dest}")
    raise typer.Exit()


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
            "health": ("control.health", "app", "Unified all-systems-green check: config, WPS, Quabos, gRPC, containers."),
            "stop": ("control.stop", "app", "Stop and finish the current recording run."),
            "xfr": ("control.transfer.cli", "app", "Inspect and manage the background transfer queue."),
            "session-start": ("control.session_start", "app", "Initialize hardware/power for an observing session."),
            "session-stop": ("control.session_stop", "app", "Gracefully power down and terminate a session."),
            # System commands
            "show": ("control.tools.show_cli", "app", "Inspect and visualize system state (sci data, pff)."),
            "paths": ("control.tools.show_cli", "show_paths", "Display resolved system paths and environment overrides."),
            "env": ("control.tools.show_cli", "show_env", "Show the resolved pseti environment variables."),
            "test": ("ci.test_cli", "app", "Unified PSETI testing suite (lint, sw, hw, pff)."),
            "grpc": ("panoseti_grpc.cli", "app", "gRPC service operations (health, reflection, etc)."),
            "admin": ("control.admin.cli", "app", "Admin/Deployment tools for remote nodes."),
        }
        # Explicit order to ensure consistent UX regardless of mapping insertion order
        command_order = [
            "power", "uids", "cfg", "val", "start", "stat", "health", "stop", "xfr",
            "session-start", "session-stop", "show", "paths", "env", "test", "grpc", "admin"
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
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree and exit.", callback=display_tree_callback)] = False,
    no_env: Annotated[bool, typer.Option("--no-env", help="Disable automatic loading of .env files.")] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Print the installed pseti-ctl package version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
    env_template: Annotated[
        bool,
        typer.Option(
            "--env-template",
            help="Copy the packaged .env.example to ./.env_pseti_<timestamp> and exit.",
            callback=_env_template_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """PSETI Control Plane."""
    pass

if __name__ == "__main__":
    app()
