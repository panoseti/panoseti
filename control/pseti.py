import typer

from ci.qa import app as test_app

app = typer.Typer(
    help="PANOSETI Observatory Control CLI",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Mount sub-apps
app.add_typer(test_app, name="test")

if __name__ == "__main__":
    app()
