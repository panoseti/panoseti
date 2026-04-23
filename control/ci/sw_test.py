import asyncio
from typing import Annotated

import typer
from qa_utils import QA_TOML_PATH, TestRunner

app = typer.Typer(
    help="PSETI Quality Assurance & Testing Suite",

    no_args_is_help=True,
    rich_markup_mode="rich",
)

@app.callback()
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", "--no-teardown", help="Bypass container teardown for debugging."),
    no_build: bool = typer.Option(False, "--no-build", help="Do not attempt to build images, use existing ones."),
    tool: str = typer.Option("docker", "--tool", help="Container tool to use (docker or podman).")
):
    ctx.obj = TestRunner(QA_TOML_PATH)
    ctx.obj.no_teardown = debug
    ctx.obj.no_build = no_build
    ctx.obj.container_tool = tool

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def lint(
    ctx: typer.Context,
    targets: Annotated[str, typer.Argument(help="Scope to lint: 'ruff', 'mypy', or 'all'")] = "all",
):
    """Run linters [ruff/mypy args...]"""
    ok = asyncio.run(ctx.obj.run_suite("lint", target=targets, extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def unit(ctx: typer.Context, jobs: int | None = typer.Option(None, "--jobs", "-j")):
    """Run unit tests [-j N] [pytest args...]"""
    ok = asyncio.run(ctx.obj.run_suite("unit", jobs=jobs, extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def integration(ctx: typer.Context):
    """Run integration tests [pytest args...]"""
    ok = asyncio.run(ctx.obj.run_suite("integration", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def chaos(ctx: typer.Context):
    """Run chaos scenario tests [pytest args...]"""
    ok = asyncio.run(ctx.obj.run_suite("chaos", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def structural(ctx: typer.Context):
    """Run structural tests [pytest args...]"""
    ok = asyncio.run(ctx.obj.run_suite("structural", extra_args=ctx.args))
    if not ok:
        raise typer.Exit(code=1)

@app.command(name="build")
def build_images():
    """Rebuild all test images"""
    runner = TestRunner(QA_TOML_PATH)
    asyncio.run(runner.build_all())

@app.command()
def cleanup():
    """Tear down all test containers and volumes"""
    runner = TestRunner(QA_TOML_PATH)
    asyncio.run(runner.cleanup_all())

@app.command(name="all", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def all_tests(ctx: typer.Context, jobs: int | None = typer.Option(None, "--jobs", "-j")):
    """Run the full testing suite"""
    suites = ["lint", "unit", "structural", "integration", "chaos"]
    success = True
    for s in suites:
        ok = asyncio.run(ctx.obj.run_suite(s, jobs=jobs, extra_args=ctx.args))
        success = success and ok
    if not success:
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
