"""Port-forwarding-aware rsync command builder for DAQ node transfers."""
from __future__ import annotations

import pathlib

from control.utils.pydantic_config_models import TransferNodeSpec
from control.utils.util import hp_stdout_prefix, pss_prefix, ssh_options


def build_rsync_cmd(
    node: TransferNodeSpec,
    run_name: str,
    head_run_dir: str | pathlib.Path,
) -> list[str]:
    """Build an rsync command for a single DAQ node's run directory.

    Pulls per-run stdout/snapshot files and all per-module PFF data from the
    remote DAQ node into the matching head-node run directory.  When the node's
    ``port_forwarding`` config has ``status=True``, the rsync tunnel is routed
    through the gateway.

    Args:
        node: A ``TransferNodeSpec`` describing the remote DAQ node.
        run_name: Name of the run directory to transfer.
        head_run_dir: Absolute path to the destination run directory on the
            head node.

    Returns:
        A list of strings forming a complete rsync invocation, ready to pass
        to ``subprocess.run``.

    Raises:
        ValueError: If the port-forwarding port is outside [1024, 65535].
    """
    pf = node.port_forwarding
    use_pf = pf is not None and pf.status

    cmd: list[str] = ["rsync", "-aP", "--info=progress2", "--partial-dir=.rsync-partial"]
    ssh_base = list(ssh_options)  # copy to avoid mutation

    if use_pf and pf is not None:
        port = pf.port
        if port is None or not (1024 <= port <= 65535):
            raise ValueError(
                f"Port-forwarding port {port!r} is out of valid range [1024, 65535]"
            )
        full_ssh_cmd = f"ssh -p {port} {' '.join(ssh_base)}"
        cmd += ["-e", full_ssh_cmd]
        host = f"{node.username}@{pf.gw_ip}"
    else:
        cmd += ["-e", f"ssh {' '.join(ssh_base)}"]
        host = f"{node.username}@{node.ip_addr}"

    head_run = str(head_run_dir)
    cmd += [
        f"{host}:{node.data_dir}/{run_name}/{hp_stdout_prefix}*",
        f"{host}:{node.data_dir}/{run_name}/{pss_prefix}*",
        *[f"{host}:{node.data_dir}/module_{m}/{run_name}/" for m in node.module_ids],
        head_run,
    ]
    return cmd
