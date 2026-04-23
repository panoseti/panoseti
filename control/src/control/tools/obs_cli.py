import importlib
import sys
from pathlib import Path
from typing import Any

import click
import typer
import typer.core

class ObsLazyGroup(typer.core.TyperGroup):
    """
    Lazy-loading group for PSETI Observatory operations.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lazy_mapping = {
            "start": ("control.start", "app", "Start a new recording run."),
            "stop": ("control.stop", "app", "Stop and finish the current recording run."),
            "status": ("control.status", "app", "Show observatory health and acquisition status."),
            "session-start": ("control.session_start", "app", "Initialize hardware/power for an observing session."),
            "session-stop": ("control.session_stop", "app", "Gracefully power down and terminate a session."),
            "uids": ("control.get_uids", "app", "Scan and record Quabo hardware UIDs."),
            "config": ("control.config", "app", "Configure observatory hardware and daemons."),
            "power": ("control.power", "app", "Control Quabo power via WPS."),
            "val": ("control.config", "validate_app", "Configuration and topology validation tools."),
        }

    def list_commands(self, ctx: click.Context) -> list[str]:
        base_cmds = super().list_commands(ctx)
        return sorted(set(base_cmds) | set(self.lazy_mapping.keys()))

    def get_command(self, ctx: click.Context, name: str) -> click.Command | None:
        cmd = super().get_command(ctx, name)
        if cmd is not None:
            return cmd

        if name in self.lazy_mapping:
            module_path, attr_name, help_str = self.lazy_mapping[name]
            
            # Help mode optimization
            is_help_mode = any(arg in sys.argv for arg in ["--help", "-h"])
            is_targeting_this = (name in sys.argv)
            if is_help_mode and not is_targeting_this and not getattr(ctx, "resilient_parsing", False):
                return click.Command(name, help=help_str)

            try:
                mod = importlib.import_module(module_path)
                obj = getattr(mod, attr_name)
                
                if isinstance(obj, typer.Typer):
                    click_cmd = typer.main.get_command(obj)
                else:
                    temp_app = typer.Typer()
                    temp_app.command(name=name, help=help_str)(obj)
                    click_cmd = typer.main.get_command(temp_app)

                # Promote single-command groups
                if isinstance(click_cmd, click.Group):
                    command_names = click_cmd.list_commands(ctx)
                    if len(command_names) == 1:
                        actual_cmd = click_cmd.get_command(ctx, command_names[0])
                        if actual_cmd:
                            if not actual_cmd.help: actual_cmd.help = click_cmd.help
                            actual_cmd.name = name
                            return actual_cmd

                click_cmd.name = name
                if not click_cmd.help: click_cmd.help = help_str
                return click_cmd
            except Exception as e:
                click.secho(f"Error loading command '{name}': {e}", fg="red", err=True)
                return None
        return None

app = typer.Typer(
    cls=ObsLazyGroup,
    help="Observatory operations (Start/Stop, Power, Config, Validation).",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

@app.callback()
def main() -> None:
    """PSETI Observatory Control."""
    pass
