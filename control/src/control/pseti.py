import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from panoseti_grpc.util.cli import BaseLazyGroup, display_tree_callback


def pseti_path_injector(name: str) -> None:
    """Inject paths for PSETI development and CI environments."""
    # Special handling for QA suites which live in control/ci/
    if name == "test":
        ci_path = str(Path(__file__).parent.parent.parent / "ci")
        if Path(ci_path).exists() and ci_path not in sys.path:
            sys.path.insert(0, ci_path)
    
    # Support dev environments where panoseti_grpc is adjacent
    if name == "grpc":
        grpc_path = str(Path(__file__).parent.parent.parent.parent / "grpc" / "src")
        if Path(grpc_path).exists() and grpc_path not in sys.path:
            sys.path.insert(0, grpc_path)


class PanoLazyGroup(BaseLazyGroup):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        lazy_mapping = {
            "obs": ("control.tools.obs_cli", "app", "Observatory operations (Start/Stop, Power, Config)."),
            "test": ("test_cli", "app", "Unified PSETI testing suite (lint, sw, hw)."),
            "grpc": ("panoseti_grpc.cli", "app", "gRPC service operations (health, reflection, etc)."),
            "show": ("control.tools.show_cli", "app", "Inspect and visualize system state (paths, commands)."),
            # Root aliases for high-frequency commands
            "start": ("control.start", "app", "Alias for 'pseti obs start'."),
            "stop": ("control.stop", "app", "Alias for 'pseti obs stop'."),
            "status": ("control.status", "app", "Alias for 'pseti obs status'."),
        }
        super().__init__(*args, lazy_mapping=lazy_mapping, path_injector=pseti_path_injector, **kwargs)


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
):
    """PSETI Control Plane."""
    pass

if __name__ == "__main__":
    app()
