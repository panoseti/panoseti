import typer

from control.ci import qa
from control import (
    config
)

app = typer.Typer(
    help="PANOSETI Observatory Control CLI",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Mount sub-apps
app.add_typer(qa.app, name="test")
app.add_typer(config.app, name="config")

if __name__ == "__main__":
    app()
