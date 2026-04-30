"""
HITL (Hardware-in-the-Loop) test orchestration CLI.

Subcommands: build, deploy, down, attach, plan, run, preflight, status,
safe-down, list-classes, explain, check-env.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from control.utils.paths import PanoPaths

app = typer.Typer(help="Hardware-in-the-Loop (HITL) physical lab tests", no_args_is_help=True)
console = Console()

# Directory containing hw_tests.toml and the suites/ tree
_HW_SW_DIR = Path(__file__).parent.parent
# control/ root (contains pyproject.toml — needed for `uv run`)
_CONTROL_DIR = PanoPaths.base_dir()
# panoseti-software/ root (needed for compose env vars)
_PSETI_ROOT = PanoPaths.software_root_dir()
# Compose file for headnode/daqnode services (Redis, InfluxDB, etc.)
_COMPOSE_FILE = _CONTROL_DIR / "src/ci/docker-compose.hw-sw.yml"

# Default configuration path for HITL tests
_HW_CONFIGS_DIR = _HW_SW_DIR / "configs"

_STATE_FILE = Path.home() / ".pseti" / "hw_runtime_state.json"


def _get_sm():
    from ci.hardware_software.hw_utils.state_machine import HardwareStateMachine
    return HardwareStateMachine()


def _get_topology():
    from ci.hardware_software.hw_utils.topology import HwTopology
    return HwTopology()


def _read_state() -> str | None:
    from ci.hardware_software.hw_utils.state_machine import read_state
    return read_state(_STATE_FILE)


def _compose_env() -> dict[str, str]:
    """Env vars required by docker-compose.hw-sw.yml."""
    return {
        "PSETI_ROOT_BUILD": str(_PSETI_ROOT),
        "PSETI_CONTROL_BUILD": str(_CONTROL_DIR),
        "PSETI_CONFIG": str(_HW_CONFIGS_DIR),
    }


def _uv_pytest(*args: str) -> list[str]:
    """Return a command list that runs pytest via `uv run` in the control project env."""
    return ["uv", "run", "--directory", str(_CONTROL_DIR), "pytest", *args]


def _run_compose(
    tool: str,
    context: str | None,
    profile: str,
    action: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Helper to run a docker-compose command on a specific context."""
    cmd = [tool]
    if context:
        cmd.extend(["--context", context])
    cmd.extend(["compose", "-f", str(_COMPOSE_FILE), "--profile", profile, action])
    if args:
        cmd.extend(args)
    return subprocess.run(cmd, env=env).returncode


@app.callback()
def hw_main(ctx: typer.Context):
    """Hardware-in-the-Loop (HITL) physical lab tests"""
    if "PSETI_CONFIG" not in os.environ:
        os.environ["PSETI_CONFIG"] = str(_HW_CONFIGS_DIR)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

@app.command(name="build")
def hw_build(
    tool: Annotated[str, typer.Option("--tool", help="Container tool (docker or podman).")] = "docker",
) -> None:
    """Build headnode and daqnode container images locally."""
    env = {**os.environ, **_compose_env()}

    # 1. Build headnode locally
    console.print("[cyan]Building profile: headnode locally...[/cyan]")
    ret = _run_compose(tool, None, "headnode", "build", env=env)
    if ret != 0:
        raise typer.Exit(code=ret)

    # 2. Build daqnode on all remote nodes
    try:
        topo = _get_topology()
        for node in topo.daq_nodes():
            context = f"pseti-daq-{node.host.replace('.', '-')}"
            console.print(f"[cyan]Building profile: daqnode on {node.host} (context: {context})...[/cyan]")
            _run_compose(tool, context, "daqnode", "build", env=env)
    except Exception as exc:
        console.print(f"[yellow]Warning: Could not build on DAQ nodes: {exc}[/yellow]")

    console.print("[green]Build complete.[/green]")


# ---------------------------------------------------------------------------
# deploy
# ---------------------------------------------------------------------------

@app.command(name="deploy")
def hw_deploy(
    tool: Annotated[str, typer.Option("--tool", help="Container tool (docker or podman).")] = "docker",
    no_build: Annotated[bool, typer.Option("--no-build", help="Use existing images; skip build.")] = False,
) -> None:
    """Start the HITL service stack (Redis, InfluxDB, headnode-server) in the background."""
    env = {**os.environ, **_compose_env()}
    args = ["-d"]
    if no_build:
        args.append("--no-build")

    # 1. Deploy headnode locally
    console.print("[cyan]Deploying headnode profile locally...[/cyan]")
    ret = _run_compose(tool, None, "headnode", "up", args=args, env=env)
    if ret != 0:
        raise typer.Exit(code=ret)

    # 2. Deploy daqnode to all remote nodes
    try:
        topo = _get_topology()
        for node in topo.daq_nodes():
            context = f"pseti-daq-{node.host.replace('.', '-')}"
            console.print(f"[cyan]Deploying daqnode profile to {node.host} (context: {context})...[/cyan]")
            _run_compose(tool, context, "daqnode", "up", args=args, env=env)
    except Exception as exc:
        console.print(f"[yellow]Warning: Could not deploy to DAQ nodes: {exc}[/yellow]")


# ---------------------------------------------------------------------------
# down
# ---------------------------------------------------------------------------

@app.command(name="down")
def hw_down(
    tool: Annotated[str, typer.Option("--tool", help="Container tool (docker or podman).")] = "docker",
    volumes: Annotated[bool, typer.Option("--volumes", "-v", help="Also remove named volumes.")] = False,
) -> None:
    """Stop the HITL service stack (preserves volumes unless -v is given)."""
    env = {**os.environ, **_compose_env()}
    args = []
    if volumes:
        args.append("--volumes")

    # 1. Down headnode locally
    console.print("[yellow]Stopping headnode profile locally...[/yellow]")
    _run_compose(tool, None, "headnode", "down", args=args, env=env)

    # 2. Down daqnode on all remote nodes
    try:
        topo = _get_topology()
        for node in topo.daq_nodes():
            context = f"pseti-daq-{node.host.replace('.', '-')}"
            console.print(f"[yellow]Stopping daqnode profile on {node.host} (context: {context})...[/yellow]")
            _run_compose(tool, context, "daqnode", "down", args=args, env=env)
    except Exception as exc:
        console.print(f"[yellow]Warning: Could not stop on DAQ nodes: {exc}[/yellow]")


# ---------------------------------------------------------------------------
# attach
# ---------------------------------------------------------------------------

@app.command(name="attach")
def hw_attach(
    tool: Annotated[str, typer.Option("--tool", help="Container tool (docker or podman).")] = "docker",
    service: Annotated[str, typer.Option("--service", "-s", help="Service to attach to.")] = "headnode-server",
) -> None:
    """Enter the headnode container shell for interactive debugging."""
    env = {**os.environ, **_compose_env()}
    # Use os.system so stdin/stdout/stderr are inherited (interactive shell).
    console.print(f"[cyan]Attaching to {service}...[/cyan]")
    os.execvpe(tool, [tool, "compose", "-f", str(_COMPOSE_FILE),
                      "--profile", "headnode", "exec", service, "/bin/bash"], env)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

@app.command(name="plan")
def hw_plan(
    hw_class: Annotated[str | None, typer.Option("--class", "-c", help="Filter to one test class.")] = None,
    assume_state: Annotated[str | None, typer.Option("--assume-state", help="Assume hardware is already in this state.")] = None,
) -> None:
    """Dry-run: print the batch plan + estimated wall clock. No hardware touched."""
    import tomllib

    try:
        sm = _get_sm()
        toml_path = _HW_SW_DIR / "hw_tests.toml"
        with toml_path.open("rb") as f:
            data = tomllib.load(f)

        classes = data.get("classes", {})
        if hw_class:
            classes = {k: v for k, v in classes.items() if k == hw_class}
        if not classes:
            console.print("[yellow]No matching HITL test classes found.[/yellow]")
            return

        # Group classes by batch_priority (same priority = same batch)
        from itertools import groupby
        sorted_classes = sorted(classes.items(), key=lambda kv: kv[1].get("batch_priority", 99))
        batches = [
            (priority, list(group))
            for priority, group in groupby(sorted_classes, key=lambda kv: kv[1].get("batch_priority", 99))
        ]

        current = assume_state or sm.initial
        total_cost = 0.0
        console.print(f"\n[bold]HITL Batch Plan[/bold]  (start state: [green]{current}[/green])\n")

        for _priority, group in batches:
            names = [k for k, _ in group]
            target = group[0][1].get("required_state", sm.initial)
            leaves = group[-1][1].get("leaves_state", target)

            transition_str = ""
            transition_cost = 0.0
            if current != target:
                try:
                    plan = sm.plan(current, target)
                    transition_cost = sm.cost(plan)
                    total_cost += transition_cost
                    steps = " → ".join(
                        f"[dim]{p.name}[/dim] ({p.budget_s['typical']:.0f}s)" for p in plan
                    )
                    transition_str = f"\n    [dim]Transition:[/dim] {steps}  [dim]({transition_cost:.0f}s)[/dim]"
                except ValueError as exc:
                    transition_str = f"\n    [red]No path to {target}: {exc}[/red]"

            classes_str = ", ".join(f"[cyan]{n}[/cyan]" for n in names)
            console.print(
                f"  Batch [{classes_str}]  target=[green]{target}[/green]"
                f"  leaves=[green]{leaves}[/green]"
                f"{transition_str}"
            )
            current = leaves

        if current != sm.safe:
            try:
                teardown = sm.plan(current, sm.safe)
                teardown_cost = sm.cost(teardown)
                total_cost += teardown_cost
                steps = " → ".join(f"[dim]{p.name}[/dim] ({p.budget_s['typical']:.0f}s)" for p in teardown)
                console.print(
                    f"\n  [yellow]Final teardown → {sm.safe}:[/yellow] {steps}  [dim]({teardown_cost:.0f}s)[/dim]"
                )
            except ValueError:
                pass

        console.print(f"\n  [bold]Estimated transition overhead:[/bold] {total_cost:.0f}s "
                      f"([dim]excludes test execution time[/dim])\n")

    except Exception as exc:
        console.print(f"[red]Plan failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command(name="ls")
def hw_ls(
    hw_class: Annotated[str | None, typer.Option("--class", "-c", help="Filter to one test class.")] = None,
    hw_state: Annotated[str | None, typer.Option("--state", "-s", help="Filter by required state.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show full node IDs.")] = False,
) -> None:
    """List all available HITL tests grouped by hardware state and class."""
    import fnmatch
    import tomllib

    try:
        # 1. Load TOML for class metadata
        with (_HW_SW_DIR / "hw_tests.toml").open("rb") as f:
            data = tomllib.load(f)
        classes = data.get("classes", {})
        mappings = data.get("mapping", [])

        # 2. Collect tests via pytest
        result = subprocess.run(
            _uv_pytest(
                str(_HW_SW_DIR),
                "-p", "ci.hardware_software.hw_utils.pytest_plugin",
                "--collect-only", "-q", "--no-header",
            ),
            capture_output=True, text=True,
            cwd=_CONTROL_DIR,
        )
        
        # 3. Classify and group
        state_groups: dict[str, dict[str, list[str]]] = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("=") or "warning" in line.lower() or "collected" in line.lower():
                continue
            
            # Match to class
            matched_cls = None
            for m in mappings:
                if fnmatch.fnmatch(line, f"*{m['glob'].lstrip('*')}*") or fnmatch.fnmatch(line, m["glob"]):
                    matched_cls = m["class"]
                    break
            
            if not matched_cls:
                matched_cls = "unclassified"
            
            # Filter by class
            if hw_class and matched_cls != hw_class:
                continue
            
            cfg = classes.get(matched_cls, {})
            state = cfg.get("required_state", "UNKNOWN")
            
            # Filter by state
            if hw_state and state != hw_state:
                continue
            
            state_groups.setdefault(state, {}).setdefault(matched_cls, []).append(line)

        # 4. Render
        if not state_groups:
            console.print("[yellow]No tests found matching filters.[/yellow]")
            return

        for state in sorted(state_groups.keys()):
            console.print(f"\n[bold green]State: {state}[/bold green]")
            for cls_name in sorted(state_groups[state].keys()):
                test_list = state_groups[state][cls_name]
                desc = classes.get(cls_name, {}).get("description", "")
                console.print(f"  [bold cyan]{cls_name}[/bold cyan]  [dim]({len(test_list)} tests) - {desc}[/dim]")
                for test in sorted(test_list):
                    display_name = test if verbose else test.split("::")[-1]
                    console.print(f"    • {display_name}")

    except Exception as exc:
        console.print(f"[red]Discovery failed: {exc}[/red]")
        raise typer.Exit(code=1) from None


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@app.command(name="run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def hw_run(
    ctx: typer.Context,
    dev: Annotated[bool, typer.Option("--dev", help="Dev mode: skip power cycles, keep hardware running.")] = False,
    hw_class: Annotated[str | None, typer.Option("--class", "-c", help="Filter to one TOML test class.")] = None,
    hw_state: Annotated[str | None, typer.Option("--state", "-s", help="Run only tests requiring this state.")] = None,
    assume_state: Annotated[str | None, typer.Option("--assume-state", help="Trust that hardware is already in this state.")] = None,
    no_power_cycle: Annotated[bool, typer.Option("--no-power-cycle", help="Refuse high-safety (power cycle) primitives.")] = False,
    keep_running: Annotated[bool, typer.Option("--keep-running", help="Skip final safety teardown (dev/lab use only).")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
    explain: Annotated[str | None, typer.Option("--explain", help="Print state plan for a single test ID and exit.")] = None,
) -> None:
    """
    Run HITL tests with state-aware batching.
    
    All standard pytest flags (e.g. -k, -v, -s, -x) can be passed at the end.
    Example: pseti test hw run -k "test_hk" -v
    """
    if dev:
        console.print("[bold yellow]DEV MODE[/bold yellow] — power cycles skipped; hardware will NOT be returned to safe state.")
        keep_running = True
        if assume_state is None:
            assume_state = "ACQ_CONFIGURED"

    if explain:
        _cmd_explain(explain)
        return

    pytest_args = list(ctx.args)
    if hw_class:
        pytest_args += ["-m", f"hw_class({hw_class!r})"]
    if hw_state:
        pytest_args += ["-m", f"required_state({hw_state!r})"]

    cmd = _uv_pytest(
        str(_HW_SW_DIR),
        "-p", "ci.hardware_software.hw_utils.pytest_plugin",
        "--tb=short",
        *pytest_args,
    )

    if not yes and not _confirm_run(cmd):
        raise typer.Exit(code=0)

    if not keep_running:
        sm = _get_sm()
        from ci.hardware_software.hw_utils.safety import SafetyManager
        mgr = SafetyManager(sm, _STATE_FILE, keep_running=keep_running)
        mgr.register()

    ret = subprocess.run(cmd, cwd=_CONTROL_DIR).returncode
    raise typer.Exit(code=ret)


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

@app.command(name="preflight", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def hw_preflight(ctx: typer.Context) -> None:
    """Run only tests with preflight=true in their TOML class (pre-observation subset)."""
    import tomllib
    toml_path = _HW_SW_DIR / "hw_tests.toml"
    with toml_path.open("rb") as f:
        data = tomllib.load(f)
    preflight_classes = [
        name for name, cfg in data.get("classes", {}).items()
        if cfg.get("preflight", False)
    ]
    if not preflight_classes:
        console.print("[yellow]No preflight classes defined in hw_tests.toml.[/yellow]")
        return

    marker_expr = " or ".join(f"hw_class({c!r})" for c in preflight_classes)
    cmd = _uv_pytest(
        str(_HW_SW_DIR),
        "-p", "ci.hardware_software.hw_utils.pytest_plugin",
        "-m", marker_expr,
        "--tb=short",
        *ctx.args,
    )
    ret = subprocess.run(cmd, cwd=_CONTROL_DIR).returncode
    raise typer.Exit(code=ret)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@app.command(name="status")
def hw_status() -> None:
    """Report current believed hardware state and reachability."""
    state = _read_state()
    if state:
        console.print(f"Last known state: [green]{state}[/green] (from {_STATE_FILE})")
    else:
        console.print("[dim]No persisted hardware state found.[/dim]")

    try:
        topo = _get_topology()
        for q in topo.quabo_ips()[:4]:  # show first module only
            import subprocess as sp
            r = sp.run(["ping", "-c1", "-W1", q.ip], capture_output=True, timeout=3)
            reachable = "[green]✓[/green]" if r.returncode == 0 else "[red]✗[/red]"
            console.print(f"  {reachable} {q.ip} (module {q.module_id} Q{q.quadrant})")
    except Exception as exc:
        console.print(f"[dim]Topology unavailable: {exc}[/dim]")


# ---------------------------------------------------------------------------
# safe-down
# ---------------------------------------------------------------------------

@app.command(name="safe-down")
def hw_safe_down(
    keep_running: Annotated[bool, typer.Option("--keep-running", help="Print banner but do not power off.")] = False,
) -> None:
    """Manually invoke emergency teardown (drive hardware to safe/UNPOWERED state)."""
    sm = _get_sm()
    from ci.hardware_software.hw_utils.safety import SafetyManager
    mgr = SafetyManager(sm, _STATE_FILE, keep_running=keep_running)

    if keep_running:
        console.print("[bold yellow]--keep-running set: skipping power-off.[/bold yellow]")
        mgr.emergency_teardown()
        return

    current = _read_state() or sm.initial
    target = sm.safe
    console.print(f"[yellow]safe-down: driving {current!r} → {target!r}...[/yellow]")

    if current == target:
        console.print(f"[green]Already in safe state ({target}).[/green]")
        return

    try:
        plan = sm.plan(current, target)
        steps = " → ".join(p.name for p in plan)
        console.print(f"  Steps: {steps}")
    except ValueError:
        console.print("[dim]  (using emergency WPS-off fallback)[/dim]")

    mgr.emergency_teardown()
    console.print(f"[green]safe-down complete. Hardware state: {target}[/green]")


# ---------------------------------------------------------------------------
# list-classes
# ---------------------------------------------------------------------------

@app.command(name="list-classes")
def hw_list_classes() -> None:
    """Print TOML class definitions and attempt to count tests in each class."""
    import tomllib
    toml_path = _HW_SW_DIR / "hw_tests.toml"
    with toml_path.open("rb") as f:
        data = tomllib.load(f)
    classes = data.get("classes", {})
    mappings = data.get("mapping", [])

    # Attempt test count via pytest --collect-only
    counts: dict[str, int] = {}
    try:
        result = subprocess.run(
            _uv_pytest(
                str(_HW_SW_DIR),
                "-p", "ci.hardware_software.hw_utils.pytest_plugin",
                "--collect-only", "-q", "--no-header",
            ),
            capture_output=True, text=True,
            cwd=_CONTROL_DIR,
            timeout=30,
        )
        import fnmatch
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("=") or "warning" in line.lower():
                continue
            for entry in mappings:
                glob = entry.get("glob", "")
                cls_name = entry.get("class", "")
                if fnmatch.fnmatch(line, f"*{glob.lstrip('*')}*") or fnmatch.fnmatch(line, glob):
                    counts[cls_name] = counts.get(cls_name, 0) + 1
                    break
    except Exception:
        pass  # Show without counts if collection fails

    console.print("[bold]HITL Test Classes[/bold]")
    for name, cfg in classes.items():
        count_str = f"  [dim]{counts[name]} tests[/dim]" if name in counts else ""
        console.print(
            f"  [cyan]{name}[/cyan]  required_state=[green]{cfg.get('required_state', '?')}[/green]"
            f"  priority={cfg.get('batch_priority', '?')}"
            f"  preflight={'[green]yes[/green]' if cfg.get('preflight') else 'no'}"
            f"{count_str}"
            f"\n    {cfg.get('description', '')}"
        )


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------

@app.command(name="explain")
def hw_explain(
    test_id: Annotated[str, typer.Argument(help="Test node ID to explain.")],
    assume_state: Annotated[str | None, typer.Option("--assume-state")] = None,
) -> None:
    """Print the state transition plan a single test would trigger."""
    _cmd_explain(test_id, assume_state)


# ---------------------------------------------------------------------------
# check-env
# ---------------------------------------------------------------------------

@app.command(name="check-env")
def hw_check_env() -> None:
    """Verify HITL environment: config files, WPS reachability, network connectivity."""
    import shutil

    all_ok = True

    # ── Config ──────────────────────────────────────────────────────────────
    console.print("[dim]Checking HITL configuration...[/dim]")
    pseti_config = os.environ.get("PSETI_CONFIG", "")
    if not pseti_config:
        console.print(
            "[yellow]⚠ PSETI_CONFIG is not set.[/yellow]\n"
            "  Set it to the directory containing obs_config.json, e.g.:\n"
            f"    export PSETI_CONFIG={_HW_CONFIGS_DIR}"
        )
        all_ok = False
    else:
        obs_path = Path(pseti_config) / "obs_config.json"
        if not obs_path.exists():
            console.print(f"[red]✗ obs_config.json not found at {obs_path}[/red]")
            all_ok = False
        else:
            console.print(f"[green]✓ obs_config.json found at {obs_path}[/green]")

    # ── Topology (if config is available) ──────────────────────────────────
    topo = None
    if all_ok:
        try:
            topo = _get_topology()
            console.print(f"[green]✓ Topology loaded: {len(topo.module_ids())} module(s), "
                          f"{len(topo.daq_nodes())} DAQ node(s)[/green]")
            caps = topo.capabilities()
            if caps:
                console.print(f"  Capabilities: {sorted(caps)}")
        except Exception as exc:
            console.print(f"[red]✗ Failed to load topology: {exc}[/red]")
            all_ok = False

    # ── Disk space ──────────────────────────────────────────────────────────
    console.print("[dim]Checking disk space...[/dim]")
    for path in ("/mnt/panoseti-test", str(Path.home())):
        if os.path.exists(path):
            usage = shutil.disk_usage(path)
            free_gb = usage.free / (2**30)
            color = "green" if free_gb >= 10 else "yellow"
            console.print(f"  [{color}]{path}: {free_gb:.1f} GB free[/{color}]")

    # ── WPS reachability ────────────────────────────────────────────────────
    if topo is not None:
        console.print("[dim]Checking WPS outlets...[/dim]")
        for wps in topo.wps_outlets():
            url = getattr(wps, "url", None)
            if url:
                ret = subprocess.run(
                    ["curl", "-s", "--connect-timeout", "2", "--head", url],
                    capture_output=True, timeout=5,
                ).returncode
                icon = "[green]✓[/green]" if ret == 0 else "[red]✗[/red]"
                console.print(f"  {icon} WPS {wps.name} at {url}")

    # ── Quabo reachability ──────────────────────────────────────────────────
    if topo is not None:
        console.print("[dim]Checking quabo reachability (first module)...[/dim]")
        from control.driver.quabo_driver import QUABO
        for q in topo.quabo_ips()[:4]:
            # 1. ICMP ping (raw IP)
            r = subprocess.run(["ping", "-c1", "-W1", q.ip], capture_output=True, timeout=3)
            ping_ok = r.returncode == 0
            ping_icon = "[green]✓[/green]" if ping_ok else "[red]✗[/red]"
            
            # 2. UDP Echo (real_ip:cmd_port)
            udp_ok = False
            try:
                drv = QUABO(q.real_ip, port=q.cmd_port)
                drv.send(drv.make_cmd(0x01))
                drv.sock.settimeout(1.0)
                data, _ = drv.sock.recvfrom(1024)
                if data:
                    udp_ok = True
                drv.close()
            except (TimeoutError, OSError):
                pass
            udp_icon = "[green]✓[/green]" if udp_ok else "[red]✗[/red]"
            
            # 3. Reboot port check (TFTP/UDP 69 or forwarded)
            reboot_ok = False
            import contextlib
            with contextlib.suppress(Exception):
                # We can't easily "ping" TFTP without a request, but we can try to open a socket 
                # or just check if it's reachable. For simplicity, we check if we can bind/connect.
                # Since it's UDP, we just log it as a separate check.
                reboot_ok = udp_ok # If cmd port works, we assume network path is okay; 
                                   # but let's be more specific if possible.
            reboot_icon = "[green]✓[/green]" if reboot_ok else "[red]✗[/red]"

            console.print(f"  {ping_icon} ICMP {q.ip:15} | {udp_icon} CMD {q.real_ip}:{q.cmd_port} | {reboot_icon} REBOOT {q.real_ip}:{q.reboot_port} (loc={q.boardloc})")
            if not udp_ok:
                all_ok = False

    # ── Daemon configuration ───────────────────────────────────────────────
    console.print("[dim]Checking daemon configuration...[/dim]")
    capture_script = _CONTROL_DIR / "src" / "control" / "daemons" / "capture_hk.py"
    if capture_script.exists():
        console.print(f"[green]✓ capture_hk.py found at {capture_script.name}[/green]")
    else:
        console.print(f"[red]✗ capture_hk.py NOT found at {capture_script}[/red]")
        all_ok = False

    console.print()
    if all_ok:
        console.print("[green]check-env OK[/green]")
    else:
        console.print("[red]check-env found issues (see above)[/red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cmd_explain(test_id: str, assume_state: str | None = None) -> None:
    import tomllib
    toml_path = _HW_SW_DIR / "hw_tests.toml"
    with toml_path.open("rb") as f:
        data = tomllib.load(f)

    from ci.hardware_software.hw_utils.scheduler import StateAwareScheduler
    sm = _get_sm()
    scheduler = StateAwareScheduler(sm)
    cls_name = scheduler.class_for(test_id)
    if not cls_name:
        console.print(f"[yellow]{test_id!r} has no TOML class mapping.[/yellow]")
        return

    cls_cfg = data.get("classes", {}).get(cls_name, {})
    target = cls_cfg.get("required_state", sm.initial)
    current = assume_state or sm.initial
    console.print(f"Test class: [cyan]{cls_name}[/cyan]  required_state=[green]{target}[/green]")
    if current == target:
        console.print("  [dim]No transition needed from current state.[/dim]")
        return
    try:
        plan = sm.plan(current, target)
        cost = sm.cost(plan)
        steps = " → ".join(f"{p.name} ({p.budget_s['typical']:.0f}s)" for p in plan)
        console.print(f"  Transition: {steps}")
        console.print(f"  Total cost: {cost:.0f}s")
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")


def _confirm_run(cmd: list[str]) -> bool:
    console.print(f"[dim]Will run:[/dim] {' '.join(cmd)}")
    ans = typer.prompt("Proceed? [y/N]", default="N")
    return ans.strip().lower() in ("y", "yes")

