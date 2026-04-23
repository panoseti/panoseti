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
        # Mapping of command/group name to (module_path, optional_attr_name)
        self.lazy_mapping = {
            "start": "control.start",
            "stop": "control.stop",
            "status": "control.status",
            "session-start": "control.session_start",
            "session-stop": "control.session_stop",
            "get-uids": "control.get_uids",
            "config": "control.config",
            "power": "control.power",
            "path": "control.paths_cli",
            "sw-test": "qa",  # Software QA suite (Unit, Integration, Chaos)
            "hw-test": ("qa", "test_hw_app"), # Hardware-Software HITL tests
            "validate": ("control.config", "validate_app"), # Sub-app within config.py
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
            entry = self.lazy_mapping[name]
            if isinstance(entry, tuple):
                module_path, attr_name = entry
            else:
                module_path, attr_name = entry, "app"
            
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
