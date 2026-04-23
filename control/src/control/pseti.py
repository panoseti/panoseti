import importlib
import sys
from pathlib import Path
from typing import Any, Annotated

import click
import typer
import typer.core


class PanoLazyGroup(typer.core.TyperGroup):
    """
    Custom Click Group that lazy-loads commands from other modules.
    Ensures that heavy dependencies (like Protobuf or Rich) aren't loaded
    until a specific command is actually executed.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Mapping of command/group name -> (module_path, attr_name, help_string)
        self.lazy_mapping = {
            "obs": ("control.tools.obs_cli", "app", "Observatory operations (Start/Stop, Power, Config)."),
            "test": ("test_cli", "app", "Unified PSETI testing suite (lint, sw, hw)."),
            "grpc": ("panoseti_grpc.cli", "app", "gRPC service operations (health, reflection, etc)."),
            "show": ("control.tools.show_cli", "app", "Inspect and visualize system state (paths, commands)."),
            # Root aliases for high-frequency commands
            "start": ("control.start", "app", "Alias for 'pseti obs start'."),
            "stop": ("control.stop", "app", "Alias for 'pseti obs stop'."),
            "status": ("control.status", "app", "Alias for 'pseti obs status'."),
        }

    def list_commands(self, ctx: click.Context) -> list[str]:
        base_cmds = super().list_commands(ctx)
        return sorted(set(base_cmds) | set(self.lazy_mapping.keys()))

    def get_command(self, ctx: click.Context, name: str) -> click.Command | None:
        # 1. Try standard command
        cmd = super().get_command(ctx, name)
        if cmd is not None:
            return cmd

        # 2. Try lazy command
        if name in self.lazy_mapping:
            module_path, attr_name, help_str = self.lazy_mapping[name]

            # Optimization: Skip loading if we just want the top-level help
            is_help_mode = any(arg in sys.argv for arg in ["--help", "-h"])
            is_targeting_this = (name in sys.argv)
            if is_help_mode and not is_targeting_this and not getattr(ctx, "resilient_parsing", False):
                return click.Command(name, help=help_str)

            # Special handling for QA suites which live in control/ci/
            if name == "test":
                ci_path = str(Path(__file__).parent.parent.parent / "ci")
                if ci_path not in sys.path:
                    sys.path.insert(0, ci_path)
            
            # Support dev environments where panoseti_grpc is adjacent
            if name == "grpc":
                grpc_path = str(Path(__file__).parent.parent.parent.parent / "grpc" / "src")
                if Path(grpc_path).exists() and grpc_path not in sys.path:
                    sys.path.insert(0, grpc_path)
            
            try:
                mod = importlib.import_module(module_path)
                if not hasattr(mod, attr_name):
                    return None
                
                # Get the Click command/group from the Typer app
                obj = getattr(mod, attr_name)
                
                if isinstance(obj, typer.Typer):
                    click_cmd = typer.main.get_command(obj)
                else:
                    # Wrap bare function in a Typer app to leverage the unwrap logic below
                    temp_app = typer.Typer()
                    temp_app.command(name=name, help=help_str)(obj)
                    click_cmd = typer.main.get_command(temp_app)

                # Promote single-command groups to actual commands
                if isinstance(click_cmd, click.Group):
                    command_names = click_cmd.list_commands(ctx)
                    if len(command_names) == 1:
                        actual_cmd = click_cmd.get_command(ctx, command_names[0])
                        if actual_cmd:
                            if not actual_cmd.help:
                                actual_cmd.help = click_cmd.help
                            actual_cmd.name = name
                            return actual_cmd

                click_cmd.name = name
                if not click_cmd.help:
                    click_cmd.help = help_str
                return click_cmd
                
            except Exception as e:
                click.secho(f"Error loading command '{name}': {e}", fg="red", err=True)
                return None
        return None

app = typer.Typer(
    cls=PanoLazyGroup,
    help="PSETI Observatory Control CLI",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

def display_tree_callback(value: bool):
    if value:
        from control.tools.show_cli import show_commands
        # Create a dummy Typer Context to satisfy show_commands
        # or just invoke it directly with an empty context walk
        from typer import Context
        import click
        ctx = click.Context(typer.main.get_command(app))
        # We need a proper Typer context walk to reach the tree logic
        # But we can just use show_commands logic directly
        from rich.console import Console
        from rich.tree import Tree
        from control.tools.show_cli import _walk_commands
        console = Console()
        root_tree = Tree("[bold reverse] PSETI CLI Structure [/]")
        _walk_commands(ctx.command, root_tree)
        console.print("\n", root_tree, "\n")
        raise typer.Exit()

@app.callback()
def main_callback(
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree and exit.", callback=display_tree_callback)] = False
):
    """PSETI Control Plane."""
    pass

if __name__ == "__main__":
    app()
