import typer

from control import config, power, session_start, session_stop, start, status, stop
from control.ci import qa

app = typer.Typer(
    help="PANOSETI Observatory Control CLI",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Mount sub-apps
app.add_typer(qa.app, name="test")
app.add_typer(config.app, name="config")
app.add_typer(power.app, name="power")

# Register top-level commands
app.command(name="start")(start.main)
app.command(name="stop")(stop.main)
app.command(name="status")(status.main)
app.command(name="session-start")(session_start.main)
app.command(name="session-stop")(session_stop.main)

if __name__ == "__main__":
    app()
