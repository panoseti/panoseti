import importlib
import sys
from pathlib import Path

import click
import typer
import typer.core


class PanoLazyGroup(typer.core.TyperGroup):
    """
    Custom Click Group that lazy-loads commands from other modules.
    This eliminates the need to duplicate argument signatures in this main entry point.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mapping of command/group name to (module_path, optional_attr_name, help_string)
        self.lazy_mapping = {
            "start": ("control.start", "app", "Start a new recording run."),
            "stop": ("control.stop", "app", "Stop and finish the current recording run."),
            "status": ("control.status", "app", "Show control plane status."),
            "session-start": ("control.session_start", "app", "Initialize hardware/power for an observing session."),
            "session-stop": ("control.session_stop", "app", "Gracefully terminate a session."),
            "get-uids": ("control.get_uids", "app", "Scan and record Quabo hardware UIDs."),
            "config": ("control.config", "app", "Configure observatory hardware and daemons."),
            "power": ("control.power", "app", "Control Quabo power via WPS."),
            "path": ("control.tools.paths_cli", "app", "Manage PANOSETI directory paths."),
            "sw-test": ("sw_test", "app", "Software Quality Assurance & Testing Suite."),
            "hw-test": ("hw_test", "app", "Hardware-Software (HITL) tests."),
            "validate": ("control.config", "validate_app", "Configuration and topology validation tools."),
        }

    def list_commands(self, ctx: click.Context) -> list[str]:
        base_cmds = super().list_commands(ctx)
        return sorted(set(base_cmds) | set(self.lazy_mapping.keys()))

    def get_command(self, ctx: click.Context, name: str) -> click.Command | None:
        # 1. Try to get standard command (already registered via @app.command or add_typer)
        cmd = super().get_command(ctx, name)
        if cmd is not None:
            return cmd
        
        # 2. Try to get lazy command
        if name in self.lazy_mapping:
            module_path, attr_name, help_str = self.lazy_mapping[name]
            
            # Optimization: If we are likely just listing commands for the parent's --help,
            # return a dummy command with the help string to avoid loading the module.
            # We load the module only if:
            # - We are explicitly asked for help for THIS command (pseti cmd --help)
            # - We are executing THIS command (pseti cmd ...)
            # - Click is in resilient parsing mode (completion)
            # - We are NOT in help mode (programmatic/test usage)
            is_help_mode = any(arg in sys.argv for arg in ["--help", "-h"])
            is_targeting_this = (name in sys.argv)
            if is_help_mode and not is_targeting_this and not getattr(ctx, "resilient_parsing", False):
                return click.Command(name, help=help_str)

            # Special handling for QA suites which live in control/ci/
            if name in ["sw-test", "hw-test"]:
                ci_path = str(Path(__file__).parent.parent.parent / "ci")
                if ci_path not in sys.path:
                    sys.path.insert(0, ci_path)
            
            try:
                mod = importlib.import_module(module_path)
                if not hasattr(mod, attr_name):
                    return None
                
                # Get the Click command/group from the Typer app
                obj = getattr(mod, attr_name)
                click_cmd = typer.main.get_command(obj) if isinstance(obj, typer.Typer) else obj
                
                # Unwrap: If the module app has exactly one command (e.g., 'main' in start.py),
                # promote it so 'pseti start --help' works directly instead of 'pseti start main --help'.
                if isinstance(click_cmd, click.Group):
                    command_names = click_cmd.list_commands(ctx)
                    if len(command_names) == 1:
                        actual_cmd = click_cmd.get_command(ctx, command_names[0])
                        if actual_cmd:
                            # Inherit help from the group if the command lacks it
                            if not actual_cmd.help:
                                actual_cmd.help = click_cmd.help
                            actual_cmd.name = name
                            return actual_cmd
                
                click_cmd.name = name
                # Ensure the help string matches our mapping if not provided by the command
                if not click_cmd.help:
                    click_cmd.help = help_str
                return click_cmd
            except Exception as e:
                # We don't want to crash the whole CLI if one module is broken
                click.secho(f"Error loading command '{name}' from {module_path}: {e}", fg="red", err=True)
                return None
        return None

app = typer.Typer(
    cls=PanoLazyGroup,
    help="PANOSETI Observatory Control CLI",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

@app.callback()
def main_callback():
    """PANOSETI Control Plane."""
    pass

if __name__ == "__main__":
    app()
