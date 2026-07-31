import asyncio
import atexit
import contextlib
import os
import subprocess
import tempfile
from collections.abc import Coroutine
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from control.utils.paths import PanoPaths

app = typer.Typer(
    help="Admin and deployment tools for remote DAQ nodes and the head node.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()

# Env vars whose values actually change compose interpolation/behavior --
# printed alongside every command so it can be copy-pasted and re-run (or
# tweaked) verbatim, without having to reconstruct what this process
# resolved them to. Includes the gRPC port vars (see utils.util's
# resolve_grpc_port / grpc's unified_main.resolve_bind_port) since a
# misresolved port here is exactly the class of bug this list exists to
# make visible.
_PRINTABLE_ENV_KEYS = (
    "PSETI_ROOT_BUILD", "PSETI_CONFIG", "PSETI_DATA_DIR", "DAQ_DATA_DIR",
    "HEADNODE_IP", "HEADNODE_GRPC_PORT", "DAQNODE_GRPC_PORT",
    "HOST_UID", "HOST_GID", "LOCAL_UID", "LOCAL_GID",
)

# Keys actually written into the materialized --env-file (a superset of
# _PRINTABLE_ENV_KEYS: some compose-relevant vars aren't interesting enough
# to echo on every invocation but must still reach compose interpolation
# deterministically).
_ENV_FILE_KEYS = (*_PRINTABLE_ENV_KEYS, "GRPC_PORT", "DAQ_DATA_GATEWAY_HOST", "REDIS_HOST", "LOKI_URL")

# A `pseti admin` invocation can call _write_compose_env_file() several
# times (once per compose file, per node). Track them for best-effort
# cleanup on exit instead of leaking small .env files into PanoPaths.tmp_dir()
# forever -- same tempfile-cleanup pattern as health.py's _check_quabo_tftp.
_tmp_env_files: list[Path] = []


@atexit.register
def _cleanup_tmp_env_files() -> None:
    for p in _tmp_env_files:
        with contextlib.suppress(OSError):
            p.unlink()


def _write_compose_env_file(env: dict[str, str]) -> Path:
    """Materialize the resolved env subset to a file compose reads via --env-file.

    Passing env= to subprocess.run already makes these values reach compose
    interpolation via inheritance, but docker compose *also* auto-reads a
    `.env` file from its own project directory (which defaults to the
    caller's CWD, not necessarily the repo root) -- if `pseti admin` is run
    from a directory that happens to have an unrelated `.env`, that could
    silently compete with what we intend to pass. An explicit --env-file
    on every invocation makes interpolation deterministic regardless of
    CWD, and is also how a PSETI_ENV_FILE-selected dotfile's values reach
    compose without a hand rebuild of every var here (they're already in
    os.environ by the time this runs, via env_loader.load_pseti_env()).
    """
    tmp_dir = PanoPaths.tmp_dir()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="pseti-admin-", suffix=".env", dir=str(tmp_dir))
    with os.fdopen(fd, "w") as f:
        for key in _ENV_FILE_KEYS:
            if key in env:
                f.write(f"{key}={env[key]}\n")
    result = Path(path)
    _tmp_env_files.append(result)
    return result


def run_cmd(host: str, cmd: list[str], env: dict[str, str] | None = None, quiet: bool = False) -> bool:
    """Run a shell command, printing the full reproducible invocation first."""
    if not quiet:
        if env:
            shown = " ".join(f"{k}={env[k]}" for k in _PRINTABLE_ENV_KEYS if k in env)
            if shown:
                console.print(f"[[bold cyan]{host}[/bold cyan]] {shown} \\")
        console.print(f"[[bold cyan]{host}[/bold cyan]] Executing: {' '.join(cmd)}")

    # We use subprocess.run so output streams nicely
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        if not quiet:
            console.print(f"[[bold red]{host}[/bold red]] Command failed with exit code {result.returncode}")
        return False

    if not quiet:
        console.print(f"[[bold green]{host}[/bold green]] Command succeeded.")
    return True


async def run_cmd_async(host: str, cmd: list[str], env: dict[str, str] | None = None) -> bool:
    """Async counterpart to run_cmd(), for running several nodes' jobs concurrently.

    `deploy`/`build` used to loop over nodes and `run_cmd()` (subprocess.run,
    blocking) each one in turn -- correct but O(N) wall-clock in the number
    of nodes, since a slow/remote docker --context build for one node blocks
    every node after it in the list. subprocess.run's rationale ("output
    streams nicely") stops applying once N nodes run at once: N processes
    writing straight to the same terminal fd would interleave mid-line into
    unreadable byte-soup.

    Instead, capture each subprocess's merged stdout+stderr via a pipe and
    print it line-by-line, every line prefixed with [host] -- the same
    convention docker compose/ansible/pm2 use for multiplexed job output.
    This does NOT need an explicit lock: asyncio is single-threaded
    cooperative concurrency, and console.print() contains no `await`, so it
    always runs to completion before the next task gets scheduled -- lines
    from different nodes interleave with each other (expected, desired),
    never *within* a line.
    """
    if env:
        shown = " ".join(f"{k}={env[k]}" for k in _PRINTABLE_ENV_KEYS if k in env)
        if shown:
            console.print(f"[[bold cyan]{host}[/bold cyan]] {shown} \\")
    console.print(f"[[bold cyan]{host}[/bold cyan]] Executing: {' '.join(cmd)}")

    proc = await asyncio.create_subprocess_exec(
        *cmd, env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    async for raw_line in proc.stdout:
        line = raw_line.decode(errors="replace").rstrip("\n")
        console.print(f"[[bold cyan]{host}[/bold cyan]] {line}")
    returncode = await proc.wait()

    if returncode != 0:
        console.print(f"[[bold red]{host}[/bold red]] Command failed with exit code {returncode}")
        return False
    console.print(f"[[bold green]{host}[/bold green]] Command succeeded.")
    return True


def _daq_compose_env() -> dict[str, str]:
    """Build the env dict shared by every DAQ-node compose invocation.

    Previously duplicated inline in deploy_node() and build(), and simply
    omitted (env=None, relying on bare subprocess inheritance) in down()
    and status() -- which meant those two never got LOCAL_UID/LOCAL_GID and
    never printed the reproducible env header run_cmd() prints for
    deploy/build, making a `pseti admin down`/`status` silently harder to
    debug than a `deploy` for the exact same node.
    """
    env = os.environ.copy()
    env["LOCAL_UID"] = str(os.getuid())
    env["LOCAL_GID"] = str(os.getgid())
    return env


def _compose_prefix(context: str | None, project_name: str, compose_file: Path, env: dict[str, str]) -> list[str]:
    """Build the shared `docker [--context X] compose -p P --env-file F -f FILE` prefix.

    Centralizing this is what guarantees --env-file is never forgotten on
    a new call site the way plain env= was on down()/status() -- see
    _write_compose_env_file()'s docstring for why --env-file (not just
    env=) matters.
    """
    cmd = ["docker"]
    if context is not None:
        cmd += ["--context", context]
    env_file = _write_compose_env_file(env)
    cmd += ["compose", "-p", project_name, "--env-file", str(env_file), "-f", str(compose_file)]
    return cmd


def get_docker_context_for_node(host: str) -> str:
    from control.utils.config_file import get_daq_config
    daq_config = get_daq_config()
    try:
        node = daq_config.get_node_by_ip(host)
        if node.docker_context:
            return node.docker_context
    except Exception:
        pass
    return f"pseti-daq-{host.replace('.', '-')}"

async def _run_node_job(label: str, coro: Coroutine[Any, Any, bool | None], results: dict[str, bool]) -> None:
    """Run one node's build/deploy job under a TaskGroup, recording its outcome.

    Catches exceptions here (rather than letting them propagate to the
    TaskGroup) so one node's failure -- a bad SSH host, a network blip --
    doesn't cancel every other node's already-in-flight job. This is the
    "best-effort fan-out" shape (see grpc/CLAUDE.md's TaskGroup guidance):
    concurrent, but every node gets to finish and report its own outcome
    independently. `results` is shared across all concurrent tasks but
    each task only ever writes its own `label` key, so this needs no lock.
    """
    try:
        ok = await coro
        results[label] = ok is not False
    except Exception as exc:
        console.print(f"[[bold red]{label}[/bold red]] Job raised an exception: {exc}")
        results[label] = False


def _print_job_summary(action: str, results: dict[str, bool]) -> None:
    """Print a compact pass/fail table after a concurrent multi-node run.

    Individual nodes' output above is interleaved (by design -- see
    run_cmd_async), so a final summary is what actually answers "did
    everything succeed" without having to scroll back through it.
    """
    console.print(f"\n[bold]{action} summary:[/bold]")
    for label, ok in results.items():
        # [[...]] (not [...]) -- a bare f"[{label}]" gets swallowed as
        # unrecognized Rich markup for labels that look like a style/tag
        # name (e.g. "headnode" -- silently prints as "", no error, no
        # brackets); labels with dots (e.g. "192.168.0.228") happen not to
        # parse as markup and are unaffected, which is what made this easy
        # to miss. [[ / ]] is Rich's literal-bracket escape; matches the
        # convention already used everywhere else in this file (run_cmd()).
        status = "[bold green]OK[/bold green]" if ok else "[bold red]FAILED[/bold red]"
        console.print(f"  [[bold cyan]{label}[/bold cyan]] {status}")
    if not all(results.values()):
        raise typer.Exit(1)


def resolve_target_nodes(nodes: str) -> tuple[list[str], bool]:
    """Expand a comma-separated node list, resolving 'all' from daq_config.json.

    Returns (daq_node_targets, include_headnode). 'headnode' is not a DAQ
    node IP -- it's pulled out and returned separately since it deploys
    locally with no docker context / SSH involved. 'all' means every DAQ
    node *and* the head node.
    """
    target_nodes = [n.strip() for n in nodes.split(",")]
    if "all" in target_nodes:
        from control.utils.config_file import get_daq_config
        daq_config = get_daq_config()
        return [str(node.ip_addr) for node in daq_config.daq_nodes], True

    include_headnode = "headnode" in target_nodes
    daq_targets = [n for n in target_nodes if n != "headnode"]
    return daq_targets, include_headnode


def get_headnode_compose_env() -> dict[str, str] | None:
    """Build the env dict required by control/deploy/docker-compose.headnode.yml.

    Returns None (after printing why) if a value that has no safe default
    -- HEADNODE_IP -- isn't set anywhere.
    """
    env = os.environ.copy()
    env["PSETI_ROOT_BUILD"] = str(PanoPaths.software_root_dir())
    env.setdefault("PSETI_CONFIG", str(PanoPaths.config_dir()))
    # Some config directories (e.g. the hardware-software test harness's
    # configs/) use a variant-swapping scheme where a file like
    # data_config.json is a symlink escaping the directory (e.g.
    # `data_config.json -> ../core_obs_configs/<variant>.json`, matching
    # compose_env.py's PSETI_CORE_OBS_CONFIGS for docker-compose.hw-sw.yml).
    # Bind-mounting only PSETI_CONFIG leaves that symlink dangling inside
    # the container (its sibling directory was never mounted). Mount the
    # sibling automatically when it exists; when it doesn't (a normal site
    # config dir with no escaping symlinks), fall back to re-mounting
    # PSETI_CONFIG itself -- a harmless no-op read-only self-mount that
    # keeps docker-compose.headnode.yml's ${PSETI_CORE_OBS_CONFIGS} mount
    # from ever resolving to an unset/missing path.
    core_obs_configs = Path(env["PSETI_CONFIG"]).parent / "core_obs_configs"
    env.setdefault(
        "PSETI_CORE_OBS_CONFIGS",
        str(core_obs_configs) if core_obs_configs.is_dir() else env["PSETI_CONFIG"],
    )
    env.setdefault("PSETI_DATA_DIR", "/mnt/panoseti-data")
    env["HOST_UID"] = str(os.getuid())
    env["HOST_GID"] = str(os.getgid())
    if "HEADNODE_IP" not in env:
        console.print(
            "[bold red][headnode][/bold red] HEADNODE_IP is not set (this machine's "
            "real IP -- required so Alloy knows where to push logs). Set it and retry, e.g.:\n"
            "    HEADNODE_IP=192.168.88.103 pseti admin deploy headnode"
        )
        return None
    return env


# Services in docker-compose.headnode.yml that a deployment may already run
# bare-metal elsewhere and want to skip starting a duplicate of here.
# loki/alloy are never optional: alloy is the log-shipping path this whole
# stack exists to provide, and loki is where it ships to. headnode-server
# (the gRPC server) is optional too -- a deployment may run it bare-metal
# (see deploy_node()'s --mode bare-metal branch) while still wanting this
# compose stack for observability only.
_HEADNODE_OPTIONAL_SERVICES = ("redis", "influxdb", "grafana", "headnode-server")


def _headnode_enabled_services() -> list[str] | None:
    """Service args to append to a headnode compose `up`/`build` command.

    Returns None (append nothing -- compose then targets every service, the
    prior/default behavior) unless PSETI_HEADNODE_DISABLE_SERVICES names one
    or more of _HEADNODE_OPTIONAL_SERVICES to skip, e.g.:
        PSETI_HEADNODE_DISABLE_SERVICES=redis,influxdb
    for a deployment that already runs Redis/InfluxDB bare-metal on this
    host. Compose only starts/builds services actually named on the command
    line when any are named at all, so the disabled ones are simply omitted
    -- `down`/`status`/`logs` are unaffected (they operate on whatever is
    actually running, not on what a prior deploy/build did or didn't start).
    """
    raw = os.environ.get("PSETI_HEADNODE_DISABLE_SERVICES", "").strip()
    if not raw:
        return None
    disabled = {s.strip() for s in raw.split(",") if s.strip()}
    unknown = disabled - set(_HEADNODE_OPTIONAL_SERVICES)
    if unknown:
        console.print(
            f"[bold red][headnode][/bold red] PSETI_HEADNODE_DISABLE_SERVICES names "
            f"unrecognized/non-optional service(s): {', '.join(sorted(unknown))}. "
            f"Only {', '.join(_HEADNODE_OPTIONAL_SERVICES)} can be disabled."
        )
        raise typer.Exit(1)
    enabled_optional = [s for s in _HEADNODE_OPTIONAL_SERVICES if s not in disabled]
    if disabled:
        console.print(
            f"[yellow][headnode][/yellow] Skipping service(s) already running "
            f"elsewhere: {', '.join(sorted(disabled))}"
        )
    return [*enabled_optional, "loki", "alloy"]


async def deploy_headnode_async(mode: str) -> bool:
    """Deploy the head node's observability + gRPC gateway stack (local machine, no SSH)."""
    if mode != "docker":
        console.print(f"[yellow][headnode][/yellow] --mode {mode} is not supported for the head node; use --mode docker.")
        return False

    env = get_headnode_compose_env()
    if env is None:
        return False

    compose_file = PanoPaths.base_dir() / "deploy" / "docker-compose.headnode.yml"
    cmd = [*_compose_prefix(None, "pseti-headnode", compose_file, env), "up", "-d", "--build"]
    enabled = _headnode_enabled_services()
    if enabled is not None:
        cmd += enabled
    return await run_cmd_async("headnode", cmd, env=env)


def status_headnode(mode: str) -> None:
    """Check the status of the head node's local compose stack."""
    if mode != "docker":
        console.print(f"[yellow][headnode][/yellow] --mode {mode} is not supported for the head node; use --mode docker.")
        return

    env = get_headnode_compose_env()
    if env is None:
        return

    compose_file = PanoPaths.base_dir() / "deploy" / "docker-compose.headnode.yml"
    cmd = [*_compose_prefix(None, "pseti-headnode", compose_file, env), "ps"]
    console.print("[[bold cyan]headnode[/bold cyan]] Status:")
    run_cmd("headnode", cmd, env=env, quiet=True)


# Remote path for the EnvironmentFile= a bare-metal node's panoseti_grpc/
# panoseti_alloy systemd units read (see grpc/scripts/setup_panoseti_grpc.sh).
# Keep in sync with that script's ENV_FILE default.
_BARE_METAL_ENV_FILE = "/etc/panoseti/grpc.env"

# Subset of the resolved env actually relevant to a bare-metal node's
# systemd units -- deliberately narrow (not the full _ENV_FILE_KEYS list):
# this file is world-readable-ish on the remote host via sudo tee, so only
# forward what start_grpc.sh / config.alloy actually consume.
_BARE_METAL_ENV_KEYS = ("HEADNODE_IP", "HEADNODE_GRPC_PORT", "DAQNODE_GRPC_PORT", "PSETI_GRPC_PROFILE")


def _resolve_bare_metal_ssh_target(host: str) -> list[str]:
    """Resolve the actual SSH args to reach a bare-metal DAQ node.

    `host` is typically the DAQ node's internal IP (daq_config.json's
    ip_addr), which for a gateway-forwarded site (see network_config.json's
    port_forwarding) is only reachable *through* the gateway, not directly.
    `--mode docker` never hits this because the docker context created for
    the node already points at the gateway's SSH endpoint -- but `--mode
    bare-metal` SSHes directly, so it needs the same port_forwarding
    resolution daq_grpc_endpoint()/build_rsync_cmd() already apply elsewhere
    (see control/src/control/transfer/rsync.py). Returns plain `[host]`
    (the prior behavior) if no forwarding is configured for this node, or if
    daq_config/network_config can't be loaded -- e.g. `host` is already a
    directly-reachable hostname/gateway address.
    """
    try:
        from control.utils.config_file import get_daq_config, get_network_config
        from control.utils.util import attach_daq_config
        daq_config = get_daq_config()
        network_config = get_network_config()
        attach_daq_config(daq_config, network_config)
        node = daq_config.get_node_by_ip(host)
        pf = node.port_forwarding
        if pf is not None and pf.status and pf.port is not None:
            return ["-p", str(pf.port), f"{node.username}@{pf.gw_ip}"]
    except Exception:
        pass
    return [host]


def _write_remote_env_file(host: str, dry_run: bool = False) -> bool:
    """Write/update the bare-metal node's systemd EnvironmentFile over SSH.

    Before this, a head-node .env port/host change had NO path onto a
    bare-metal node at all -- start_grpc.sh invoked `pseti-grpc server
    --profile X` with nothing but the operator's *interactive shell* env
    (if any), and the systemd unit's only Environment= line set
    PSETI_LOGS. This is what makes `pseti admin deploy --mode bare-metal`
    actually reconfigurable via .env instead of requiring an operator to
    SSH in and hand-edit the unit file.

    `dry_run=True` prints the command a dev would run (with interactive
    sudo) instead of executing it -- see deploy_node()'s bare-metal branch.
    """
    lines = [f"{k}={os.environ[k]}" for k in _BARE_METAL_ENV_KEYS if k in os.environ]
    if not lines:
        return True  # nothing to forward; leave whatever's already there
    content = "\n".join(lines) + "\n"
    remote_cmd = f"sudo mkdir -p $(dirname {_BARE_METAL_ENV_FILE}) && sudo tee {_BARE_METAL_ENV_FILE} >/dev/null"
    ssh_target = _resolve_bare_metal_ssh_target(host)
    if dry_run:
        console.print(f"[[bold cyan]{host}[/bold cyan]] (dry-run) Would write {_BARE_METAL_ENV_FILE}:")
        console.print(f"    ssh {' '.join(ssh_target)} '{remote_cmd}' <<'EOF'\n{content}EOF")
        return True
    result = subprocess.run(["ssh", *ssh_target, remote_cmd], input=content, text=True)
    if result.returncode != 0:
        console.print(f"[[bold red]{host}[/bold red]] Failed to write {_BARE_METAL_ENV_FILE} over SSH.")
        return False
    console.print(f"[[bold cyan]{host}[/bold cyan]] Updated {_BARE_METAL_ENV_FILE}: {', '.join(lines)}")
    return True


def _grpc_pinned_commit() -> str | None:
    """Return the exact commit SHA this checkout's grpc/ submodule has checked out.

    This is what `--mode bare-metal` installs from (via `pip install
    git+https://...@<sha>`) instead of PyPI, so a bare-metal deploy always
    matches the exact commit this head node has -- the same guarantee
    `--mode docker` gets from building the DAQ node image against local
    source (see grpc/deploy/Dockerfile.daqnode). Returns None (rather than
    raising) if the submodule directory isn't a git checkout at all, so
    callers can fail the deploy with a clear message instead of a raw
    traceback.
    """
    grpc_dir = PanoPaths.software_root_dir() / "grpc"
    try:
        result = subprocess.run(
            ["git", "-C", str(grpc_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


async def deploy_node(host: str, mode: str, dry_run: bool = False) -> bool:
    """Deploy the DAQ node software using the specified strategy.

    `dry_run` only affects `mode == "bare-metal"` -- docker mode's compose
    invocation is already printed in full by run_cmd_async() before it
    runs, so there's nothing extra a dry-run would add there.
    """

    if mode == "docker":
        # We use docker --context to build and deploy natively over SSH
        context = get_docker_context_for_node(host)

        # We assume the context is already created by the user, just like in hw-sw tests.
        # Check if the context exists. asyncio.to_thread instead of a bare
        # subprocess.run: deploy_node() now runs concurrently for every
        # node (see deploy()'s asyncio.gather), and a blocking call here
        # would stall every *other* node's task on the single shared event
        # loop for its duration.
        res = await asyncio.to_thread(
            subprocess.run, ["docker", "context", "ls", "--format", "{{.Name}}"], capture_output=True, text=True
        )
        if context not in res.stdout:
            console.print(f"[[yellow]{host}[/yellow]] Docker context '{context}' not found. Please create it first:")
            console.print(f"    docker context create {context} --docker \"host=ssh://<user>@{host}\"")
            return False

        env = _daq_compose_env()
        project_name = f"pseti-daqnode-{host.replace('.', '-')}"

        compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "docker-compose.daqnode.yml"
        cmd = [*_compose_prefix(context, project_name, compose_file, env), "up", "-d", "--build"]
        ok = await run_cmd_async(host, cmd, env=env)

        # Grafana Alloy (log shipping) is a separate host-network container on the same node.
        # Skip if this DAQ node is the head node (headnode-server stack already runs it).
        from control.utils.config_file import get_daq_config
        from control.utils.util import is_local
        is_headnode = is_local(host, get_daq_config())
        if not is_headnode:
            alloy_compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "alloy" / "docker-compose.alloy.yml"
            alloy_cmd = [*_compose_prefix(context, project_name, alloy_compose_file, env), "up", "-d", "--build"]
            ok = await run_cmd_async(host, alloy_cmd, env=env) and ok
        return ok

    elif mode == "bare-metal":
        # For bare-metal, we just SSH in, install from the exact commit
        # pinned by this checkout's grpc/ submodule gitlink, and restart
        # the service. Ensure we have the SSH key available or it will prompt.
        #
        # Was `pip install --upgrade panoseti-grpc` (PyPI) -- a published
        # release necessarily lags behind whatever's actually committed
        # here by however long since the last version bump, so a fix
        # landed in the grpc submodule was invisible to every bare-metal
        # deploy until someone remembered to cut a new release. Installing
        # from git+https pinned to _grpc_pinned_commit() means bare-metal
        # deploys always match the exact commit this head node has
        # checked out -- same guarantee `--mode docker` gets by building
        # from local source (see grpc/deploy/Dockerfile.daqnode). PyPI
        # remains the documented install path for external client-script
        # consumers (see grpc/README.md); this only changes the internal
        # deployment path.
        pinned_commit = await asyncio.to_thread(_grpc_pinned_commit)
        if pinned_commit is None:
            console.print(f"[[bold red]{host}[/bold red]] Could not resolve the grpc/ submodule's pinned commit; aborting bare-metal deploy.")
            return False
        await asyncio.to_thread(_write_remote_env_file, host, dry_run)
        # panoseti_alloy is restarted separately (not `systemctl restart A B`
        # as one call) and its failure doesn't fail the whole command --
        # setup_panoseti_grpc.sh's --no-alloy lets a node skip installing it
        # entirely, and a genuinely missing unit shouldn't be reported as a
        # failed grpc restart.
        #
        # Plain `sudo` (not a hardcoded `echo <password> | sudo -S`) -- a
        # literal password checked into source control is a real credential
        # leak, not just an inconvenience, and it also silently assumes
        # every operator's remote account uses that exact password. Restart
        # requires either passwordless sudo configured for this command on
        # the remote node, or `--dry-run` + the operator running the
        # printed commands themselves in their own interactive terminal.
        remote_cmd = (
            "source ~/miniconda3/etc/profile.d/conda.sh && conda activate grpc-py314 && "
            f"pip install --upgrade 'git+https://github.com/panoseti/panoseti_grpc.git@{pinned_commit}' && "
            "sudo systemctl restart panoseti_grpc && "
            "(sudo systemctl restart panoseti_alloy || "
            "echo 'panoseti_alloy not installed/active on this node, skipping')"
        )
        ssh_target = _resolve_bare_metal_ssh_target(host)
        if dry_run:
            console.print(
                f"[[bold cyan]{host}[/bold cyan]] (dry-run) Bare-metal deploy would run "
                f"the following on this node -- copy/paste and run manually (needs "
                f"interactive sudo):"
            )
            console.print(f"    ssh -t {' '.join(ssh_target)}")
            for step in remote_cmd.split(" && "):
                console.print(f"    {step.strip()}")
            return True
        cmd = ["ssh", "-t", *ssh_target, f"bash -c '{remote_cmd}'"]
        return await run_cmd_async(host, cmd)

    else:
        console.print(f"[red]Unknown deployment mode: {mode}[/red]")
        return False

@app.command()
def deploy(
    nodes: Annotated[str, typer.Argument(help="Comma-separated list of hostnames/IPs, 'headnode', or 'all' (every DAQ node + the head node).")],
    mode: Annotated[str, typer.Option("--mode", help="'docker' or 'bare-metal' deployment strategy.")] = "docker",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", "-n",
            help=(
                "Bare-metal only: print the per-node commands (env file write, "
                "pinned pip install, systemctl restarts) instead of running them "
                "over SSH -- for a dev to review/run manually with interactive "
                "sudo. No effect in --mode docker (its compose commands are "
                "already printed in full before they run)."
            ),
        ),
    ] = False,
) -> None:
    """Deploy the DAQ node gRPC/telemetry stack and/or the head node stack."""
    if mode not in ["docker", "bare-metal"]:
        console.print("[bold red]Error:[/] --mode must be either 'docker' or 'bare-metal'.")
        raise typer.Exit(1)

    daq_targets, include_headnode = resolve_target_nodes(nodes)

    described = (["headnode"] if include_headnode else []) + daq_targets
    action = "Dry-run" if (dry_run and mode == "bare-metal") else "Starting"
    console.print(f"[bold]{action} {mode} deployment on: {', '.join(described)} (concurrently)[/bold]")

    # Deploy every node concurrently instead of one at a time -- this used
    # to be O(N) wall-clock in the number of nodes (each node's full
    # `compose up -d --build` blocked the next). run_cmd_async() handles
    # the readability side (per-line [host]-prefixed output instead of N
    # processes' raw stdout garbling together); _run_node_job() handles
    # fault isolation (one node's SSH/build failure doesn't cancel the
    # others still in flight -- best-effort fan-out, not all-or-nothing).
    async def _run_all() -> dict[str, bool]:
        results: dict[str, bool] = {}
        async with asyncio.TaskGroup() as tg:
            if include_headnode:
                tg.create_task(_run_node_job("headnode", deploy_headnode_async(mode), results))
            for host in daq_targets:
                tg.create_task(_run_node_job(host, deploy_node(host, mode, dry_run), results))
        return results

    results = asyncio.run(_run_all())
    _print_job_summary("Deploy", results)


@app.command()
def build(
    nodes: Annotated[str, typer.Argument(help="Comma-separated list of hostnames/IPs, 'headnode', or 'all' (every DAQ node + the head node).")],
    mode: Annotated[str, typer.Option("--mode", help="'docker' or 'bare-metal' deployment strategy.")] = "docker"
) -> None:
    """Build the DAQ node gRPC/telemetry stack and/or the head node stack images."""
    if mode != "docker":
        console.print("[bold red]Error:[/] build is only supported in docker mode.")
        raise typer.Exit(1)

    daq_targets, include_headnode = resolve_target_nodes(nodes)

    described = (["headnode"] if include_headnode else []) + daq_targets
    console.print(f"[bold]Starting build on: {', '.join(described)} (concurrently)[/bold]")

    async def _run_all() -> dict[str, bool]:
        results: dict[str, bool] = {}
        async with asyncio.TaskGroup() as tg:
            if include_headnode:
                tg.create_task(_run_node_job("headnode", _build_headnode_async(), results))
            for host in daq_targets:
                tg.create_task(_run_node_job(host, _build_node_async(host), results))
        return results

    results = asyncio.run(_run_all())
    _print_job_summary("Build", results)


async def _build_headnode_async() -> bool:
    """Async build-only counterpart to deploy_headnode_async() (no `up -d`, just `build`)."""
    env = get_headnode_compose_env()
    if env is None:
        return False
    compose_file = PanoPaths.base_dir() / "deploy" / "docker-compose.headnode.yml"
    cmd = [*_compose_prefix(None, "pseti-headnode", compose_file, env), "build"]
    enabled = _headnode_enabled_services()
    if enabled is not None:
        cmd += enabled
    return await run_cmd_async("headnode", cmd, env=env)


async def _build_node_async(host: str) -> bool:
    """Async build-only counterpart to deploy_node()'s docker branch (no `up -d`, just `build`)."""
    context = get_docker_context_for_node(host)
    env = _daq_compose_env()
    project_name = f"pseti-daqnode-{host.replace('.', '-')}"

    compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "docker-compose.daqnode.yml"
    cmd = [*_compose_prefix(context, project_name, compose_file, env), "build"]
    ok = await run_cmd_async(host, cmd, env=env)

    from control.utils.config_file import get_daq_config
    from control.utils.util import is_local
    is_headnode = is_local(host, get_daq_config())
    if not is_headnode:
        alloy_compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "alloy" / "docker-compose.alloy.yml"
        alloy_cmd = [*_compose_prefix(context, project_name, alloy_compose_file, env), "build"]
        ok = await run_cmd_async(host, alloy_cmd, env=env) and ok
    return ok


@app.command()
def down(
    nodes: Annotated[str, typer.Argument(help="Comma-separated list of hostnames/IPs, 'headnode', or 'all' (every DAQ node + the head node).")],
    mode: Annotated[str, typer.Option("--mode", help="'docker' or 'bare-metal' deployment strategy.")] = "docker"
) -> None:
    """Tear down the DAQ node gRPC/telemetry stack and/or the head node stack."""
    if mode != "docker":
        console.print("[bold red]Error:[/] down is only supported in docker mode.")
        raise typer.Exit(1)

    daq_targets, include_headnode = resolve_target_nodes(nodes)

    if include_headnode:
        env = get_headnode_compose_env()
        if env is not None:
            compose_file = PanoPaths.base_dir() / "deploy" / "docker-compose.headnode.yml"
            cmd = [*_compose_prefix(None, "pseti-headnode", compose_file, env), "down"]
            run_cmd("headnode", cmd, env=env)

    for host in daq_targets:
        # Previously built with no env= at all here (down() was the one
        # command that skipped it entirely, not just the printed header --
        # see _daq_compose_env()'s docstring). subprocess inheriting the
        # bare process env happened to carry port vars through anyway, but
        # not LOCAL_UID/LOCAL_GID, and --env-file (via _compose_prefix)
        # wasn't used at all, so compose fell back to its own CWD-relative
        # .env auto-discovery instead of this process's resolved values.
        context = get_docker_context_for_node(host)
        env = _daq_compose_env()
        project_name = f"pseti-daqnode-{host.replace('.', '-')}"

        compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "docker-compose.daqnode.yml"
        cmd = [*_compose_prefix(context, project_name, compose_file, env), "down"]
        run_cmd(host, cmd, env=env)

        from control.utils.config_file import get_daq_config
        from control.utils.util import is_local
        is_headnode = is_local(host, get_daq_config())
        if not is_headnode:
            alloy_compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "alloy" / "docker-compose.alloy.yml"
            alloy_cmd = [*_compose_prefix(context, project_name, alloy_compose_file, env), "down"]
            run_cmd(host, alloy_cmd, env=env)

@app.command()
def attach(
    node: Annotated[str, typer.Argument(help="Hostname/IP of DAQ node or 'headnode'.")],
    service: Annotated[str | None, typer.Argument(help="Service to attach to (e.g. daqnode-server, headnode-server, alloy).")] = None
) -> None:
    """Open an interactive shell in a specific service container."""
    if service is None:
        service = "headnode-server" if node == "headnode" else "daqnode-server"
    resolved = _get_compose_cmd_base(node, service)
    if resolved:
        cmd_base, env = resolved
        # Without --user, `compose exec` uses the image's baked-in default
        # (root, for headnode-server/daqnode-server -- see Dockerfile.ci's
        # `headnode` stage, which ends on `USER root`). The main process
        # itself drops to `panoseti` via entrypoint.sh's `exec gosu panoseti
        # "$@"`, but a fresh exec session doesn't inherit that -- it starts
        # over from the image default. Left as root, anything the operator
        # writes through a bind mount (e.g. `pseti cfg edit` editing
        # /mnt/config) lands root-owned on the host. Match the main
        # process's user explicitly instead.
        user = f"{env.get('HOST_UID', '1000')}:{env.get('HOST_GID', '1000')}"
        # Try bash first, fallback to sh
        shell_cmd = ["/bin/sh", "-c", "if command -v bash >/dev/null; then exec bash; else exec sh; fi"]
        subprocess.run([*cmd_base, "exec", "-it", "--user", user, service, *shell_cmd], env=env)

@app.command()
def logs(
    node: Annotated[str, typer.Argument(help="Hostname/IP of DAQ node or 'headnode'.")],
    service: Annotated[str | None, typer.Argument(help="Service to view logs for.")] = None,
    follow: Annotated[bool, typer.Option("--follow", "-f", help="Follow log output.")] = True
) -> None:
    """View or tail logs for a specific service."""
    if service is None:
        service = "headnode-server" if node == "headnode" else "daqnode-server"
    resolved = _get_compose_cmd_base(node, service)
    if resolved:
        cmd_base, env = resolved
        cmd = [*cmd_base, "logs"]
        if follow:
            cmd.append("-f")
        cmd.append(service)
        subprocess.run(cmd, env=env)

@app.command(name="ls")
def list_containers(
    node: Annotated[str, typer.Argument(help="Hostname/IP of DAQ node or 'headnode'.")]
) -> None:
    """List all containers and services managed by pseti admin on a node."""
    status(node)

def _get_compose_cmd_base(node: str, service: str) -> tuple[list[str], dict[str, str]] | None:
    """Helper to get the base docker compose command + its env for a node/service.

    Returns (cmd_prefix, env) rather than mutating os.environ in place --
    the previous os.environ.update(env) approach permanently polluted this
    CLI process's own environment for the rest of its lifetime (visible to
    any later command in the same invocation, and impossible to reason
    about once two calls with different envs happened in sequence).
    """
    if node == "headnode":
        env = get_headnode_compose_env()
        if env is not None:
            compose_file = PanoPaths.base_dir() / "deploy" / "docker-compose.headnode.yml"
            return _compose_prefix(None, "pseti-headnode", compose_file, env), env
    else:
        from control.utils.config_file import get_daq_config
        try:
            daq_config = get_daq_config()
            if not any(str(n.ip_addr) == node for n in daq_config.daq_nodes):
                console.print(f"[bold red]Error:[/] Node '{node}' is not 'headnode' and was not found in daq_config.json.")
                return None
        except Exception as e:
            console.print(f"[bold red]Error:[/] Failed to load daq_config.json to validate node: {e}")
            return None

        context = get_docker_context_for_node(node)
        env = _daq_compose_env()
        project_name = f"pseti-daqnode-{node.replace('.', '-')}"
        if service == "alloy":
            compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "alloy" / "docker-compose.alloy.yml"
        else:
            compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "docker-compose.daqnode.yml"
        return _compose_prefix(context, project_name, compose_file, env), env
    return None

@app.command()
def status(
    nodes: Annotated[str, typer.Argument(help="Comma-separated list of hostnames/IPs, 'headnode', or 'all' (every DAQ node + the head node).")],
    mode: Annotated[str, typer.Option("--mode", help="'docker' or 'bare-metal' deployment strategy.")] = "docker",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", "-n",
            help="Bare-metal only: print the systemctl status command instead of running it over SSH.",
        ),
    ] = False,
) -> None:
    """Check the status of the DAQ node services and/or the head node stack."""
    daq_targets, include_headnode = resolve_target_nodes(nodes)

    if include_headnode:
        status_headnode(mode)

    for host in daq_targets:
        if mode == "docker":
            # Previously built with no env at all (relied on bare subprocess
            # inheritance + compose's own CWD-relative .env auto-discovery,
            # like down() -- see _daq_compose_env()'s docstring).
            context = get_docker_context_for_node(host)
            env = _daq_compose_env()
            project_name = f"pseti-daqnode-{host.replace('.', '-')}"

            compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "docker-compose.daqnode.yml"
            cmd = [*_compose_prefix(context, project_name, compose_file, env), "ps"]
            console.print(f"[[bold cyan]{host}[/bold cyan]] DAQ Node Status:")
            run_cmd(host, cmd, env=env, quiet=True)

            from control.utils.config_file import get_daq_config
            from control.utils.util import is_local
            is_headnode = is_local(host, get_daq_config())
            if not is_headnode:
                alloy_compose_file = PanoPaths.software_root_dir() / "grpc" / "deploy" / "alloy" / "docker-compose.alloy.yml"
                alloy_cmd = [*_compose_prefix(context, project_name, alloy_compose_file, env), "ps"]
                console.print(f"[[bold cyan]{host}[/bold cyan]] Alloy Status:")
                run_cmd(host, alloy_cmd, env=env, quiet=True)
        else:
            ssh_target = _resolve_bare_metal_ssh_target(host)
            remote_check = "systemctl is-active panoseti_grpc panoseti_alloy"
            if dry_run:
                console.print(f"[[bold cyan]{host}[/bold cyan]] (dry-run) Would check status via:")
                console.print(f"    ssh {' '.join(ssh_target)} '{remote_check}'")
                continue
            cmd = ["ssh", *ssh_target, remote_check]
            run_cmd(host, cmd, quiet=True)

if __name__ == "__main__":
    app()
