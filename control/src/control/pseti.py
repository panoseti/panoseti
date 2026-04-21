import typer

app = typer.Typer(
    help="PANOSETI Observatory Control CLI",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

@app.callback()
def main_callback():
    """PANOSETI Control Plane."""
    pass

# --- Lazy Sub-Apps (Proxy Pattern) ---
# We define the structure here so --help works, but defer heavy imports to the logic.

# 1. Test Sub-app
test_app = typer.Typer(help="Testing Suite", no_args_is_help=True)
app.add_typer(test_app, name="test")

@test_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def unit(
    ctx: typer.Context,
    jobs: int | None = typer.Option(None, "--jobs", "-j", help="Parallel workers"),
):
    """Run parallel unit tests."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "ci"))
    import qa  # type: ignore[import-untyped]
    return qa.unit(ctx, jobs)

@test_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def integration(ctx: typer.Context):
    """Run end-to-end integration tests."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "ci"))
    import qa  # type: ignore[import-untyped]
    return qa.integration(ctx)

@test_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def structural(ctx: typer.Context):
    """Run structural/topology tests."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "ci"))
    import qa  # type: ignore[import-untyped]
    return qa.structural(ctx)


@test_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def chaos(ctx: typer.Context):
    """Run TDD-forcing chaos/scenario tests."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "ci"))
    import qa  # type: ignore[import-untyped]
    return qa.chaos(ctx)

@test_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def lint(ctx: typer.Context):
    """Run Ruff and MyPy static analysis."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "ci"))
    import qa  # type: ignore[import-untyped]
    return qa.lint(ctx)

@test_app.command(name="all", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def test_all(
    ctx: typer.Context,
    jobs: int | None = typer.Option(None, "--jobs", "-j", help="Parallel workers for unit tests"),
):
    """Run the full testing suite."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "ci"))
    import qa  # type: ignore[import-untyped]
    return qa.all_tests(ctx, jobs)

@test_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def build(ctx: typer.Context):
    """Rebuild the testing Docker images."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "ci"))
    import qa  # type: ignore[import-untyped]
    return qa.build(ctx)


# 2. Config Sub-app
config_app = typer.Typer(help="Hardware configuration and management.", no_args_is_help=True)
app.add_typer(config_app, name="config")

@config_app.command()
def show():
    """Show list of domes/modules/quabos."""
    from control import config
    config.show()

@config_app.command()
def ping():
    """Ping quabos."""
    from control import config
    config.ping()

@config_app.command()
def reboot():
    """Reboot quabos."""
    from control import config
    config.reboot()

@config_app.command()
def reboot_single(ip: str = typer.Argument(..., help="Reboot a single quabo.")):
    """Reboot a single quabo."""
    from control import config
    config.reboot_single(ip)

@config_app.command()
def loads():
    """Load silver firmware in quabos."""
    from control import config
    config.loads()

@config_app.command()
def init_daq_nodes():
    """Copy software to daq nodes."""
    from control import config
    config.init_daq_nodes()

@config_app.command()
def hk_dest():
    """Set the dest IP for HK packet."""
    from control import config
    config.hk_dest()

@config_app.command()
def redis_daemons():
    """Start daemons to populate Redis with HK/GPS/WR data."""
    from control import config
    config.redis_daemons()

@config_app.command()
def stop_redis_daemons():
    """Stop the Redis population daemons."""
    from control import config
    config.stop_redis_daemons()

@config_app.command()
def permanent_daemons():
    """Start permanent system daemons."""
    from control import config
    config.permanent_daemons()

@config_app.command()
def stop_permanent_daemons():
    """Stop permanent system daemons."""
    from control import config
    config.stop_permanent_daemons()

@config_app.command()
def show_permanent_daemons():
    """Show permanent daemon status."""
    from control import config
    config.show_permanent_daemons()

@config_app.command()
def hv_on():
    """Enable detectors (High Voltage ON)."""
    from control import config
    config.hv_on()

@config_app.command()
def hv_off():
    """Disable detectors (High Voltage OFF)."""
    from control import config
    config.hv_off()

@config_app.command()
def maroc_config():
    """Configure MAROCs based on calibration files."""
    from control import config
    config.maroc_config()

@config_app.command()
def mask_config():
    """Configure masks based on data_config.json."""
    from control import config
    config.mask_config()

@config_app.command()
def calibrate_ph():
    """Run PH baseline calibration on quabos."""
    from control import config
    config.calibrate_ph()

@config_app.command()
def show_ph_baselines():
    """Show PH baseline calibration summary statistics."""
    from control import config
    config.show_ph_baselines()

@config_app.command()
def shutter_open():
    """Open all module shutters."""
    from control import config
    config.shutter_open()

@config_app.command()
def shutter_close():
    """Close all module shutters."""
    from control import config
    config.shutter_close()

@config_app.command()
def disk_space():
    """Check disk space on head and DAQ nodes."""
    from control import config
    config.disk_space()

@config_app.command()
def start_interleave():
    """Start background interleaver."""
    from control import config
    config.start_interleave()

@config_app.command()
def stop_interleave():
    """Stop background interleaver."""
    from control import config
    config.stop_interleave()

@config_app.command()
def dry_run_interleave():
    """Test the interleave schedule for 2 cycles."""
    from control import config
    config.dry_run_interleave()


# 3. Power Sub-app
power_app = typer.Typer(help="Control Quabo power via Web Power Switches (WPS).", no_args_is_help=True)
app.add_typer(power_app, name="power")

@power_app.command()
def on():
    """Turn all Quabo power switches ON."""
    from control import power
    power.on()

@power_app.command()
def off():
    """Turn all Quabo power switches OFF."""
    from control import power
    power.off()

@power_app.command()
def status():
    """Query the power state of all Quabo switches."""
    from control import power
    power.status()


# 4. Path Sub-app
path_app = typer.Typer(help="Manage and visualize PANOSETI directory paths.", no_args_is_help=True)
app.add_typer(path_app, name="path")

@path_app.command(name="show")
def show_paths():
    """Display all resolved paths and environment overrides."""
    from control import paths_cli
    paths_cli.show()

@path_app.command()
def init():
    """Create standard workspace directories if they do not exist."""
    from control import paths_cli
    paths_cli.init()

@path_app.command()
def clean():
    """Remove transient/log directories (requires confirmation)."""
    from control import paths_cli
    paths_cli.clean()


# 5. Validate Sub-app
validate_app = typer.Typer(help="Configuration and topology validation tools.", no_args_is_help=True)
app.add_typer(validate_app, name="validate")

@validate_app.callback(invoke_without_command=True)
def validate_main(ctx: typer.Context):
    """
    Validate configs. 
    By default, runs Tier-1 (Schema) and Tier-2 (Global) checks.
    """
    if ctx.invoked_subcommand is None:
        from control.utils import config_file
        passed = config_file.validate_all(check_network=False, debug=False, graph=False)
        if not passed:
            raise typer.Exit(code=1)

@validate_app.command()
def network():
    """Validate configs and perform network ping sweep."""
    from control.utils import config_file
    passed = config_file.validate_all(check_network=True)
    if not passed:
        raise typer.Exit(code=1)

@validate_app.command()
def graph():
    """Validate configs and display topology graph."""
    from control.utils import config_file
    passed = config_file.validate_all(graph=True)
    if not passed:
        raise typer.Exit(code=1)

@validate_app.command()
def debug():
    """Validate configs with verbose debug output."""
    from control.utils import config_file
    passed = config_file.validate_all(debug=True)
    if not passed:
        raise typer.Exit(code=1)

@validate_app.command(name="all")
def validate_all_modes():
    """Run all validation checks (Schema, Global, Network, Graph)."""
    from control.utils import config_file
    passed = config_file.validate_all(check_network=True, debug=True, graph=True)
    if not passed:
        raise typer.Exit(code=1)


# --- Top-Level Commands with Lazy Loading ---

@app.command(name="start")
def start_cmd(
    no_hv: bool = typer.Option(False, "--no-hv", help="Don't check HV status."),
    no_redis: bool = typer.Option(False, "--no-redis", help="Don't check Redis status."),
    no_data: bool = typer.Option(False, "--no-data", help="Don't check data collection."),
    nsecs: int = typer.Option(0, "--nsecs", help="Run for N seconds then stop."),
    stop_session: bool = typer.Option(False, "--stop-session", help="Stop session after run."),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output."),
    force_reset: bool = typer.Option(False, "--force-reset", help="Force reset of DAQ nodes."),
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation prompt."),
):
    """
    Start an observing run.

    - figure out association of quabos and DAQ nodes, based on config files
    - create \"run directories\" on head node, DAQ nodes
    """
    from control import start
    start.main(no_hv, no_redis, no_data, nsecs, stop_session, verbose, force_reset, yes)

@app.command(name="stop")
def stop_cmd(
    no_cleanup: bool = typer.Option(False, "--no-cleanup", help="Don't clean up the data files on the DAQ nodes."),
    no_collect: bool = typer.Option(False, "--no-collect", help="Don't collect the data files to the head node."),
    run: str | None = typer.Option(None, "--run", help="Stop/Cleanup specific run."),
    force_cleanup: bool = typer.Option(False, "--force-cleanup", help="Force cleanup on DAQ nodes even if hashpipe liveness is uncertain."),
    verbose: bool = typer.Option(False, "--verbose", help="Print details."),
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation prompt."),
):
    """
    Stop an observing run.

    - tell DAQs to stop recording
    - stop HK recorder process
    - tell quabos to stop sending data
    - if a run is in progress, copy data files to head and delete from DAQs
    """
    from control import stop
    stop.main(no_cleanup, no_collect, run, force_cleanup, verbose, yes)

@app.command(name="status")
def status_cmd():
    """
    Show control plane status.
    
    Checks the transactional ledger, local markers, and probes remote DAQ 
    nodes via gRPC/SSH to report on Hashpipe liveness and disk usage.
    """
    from control import status
    status.main()

@app.command(name="session-start")
def session_start_cmd(
    no_hv: bool = typer.Option(False, "--no-hv", help="Don't check HV status."),
    stage: str = typer.Option("poweron", help="The session will start from this stage: poweron, get_uids, reboot, hk_dest, start_redis, maroc_config, mask_config, calibrate_ph, show_ph_baselines.")
):
    """
    Start an observing session (power, UIDs, HV, calibration).

    - open domes (TBD)
    - power on relevant modules
    - wait for quabos to come up
    - get quabo UIDs
    - reboot quabos
    - turn on HV (using levels from quabo config files)
    - set gain params of Marocs
    - do PH baseline calibration
    - start the Redis daemons
    - copy software to DAQ nodes
    """
    from control import session_start
    session_start.main(no_hv, stage)

@app.command(name="session-stop")
def session_stop_cmd():
    """
    Gracefully terminate a session.
    
    Powers off all modules and stops background Redis daemons.
    """
    from control import session_stop
    session_stop.main()

if __name__ == "__main__":
    app()
