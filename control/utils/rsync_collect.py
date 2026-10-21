#! /usr/bin/env python3

# collect files from DAQ nodes to the head node with rsync
# at the end of a recording run.
#
# For each DAQ node, a transfer log is written to
#   <head_node_data_dir>/<run>/rsync_<ip>_<YYYYMMDD>.log
# in the same format as savedata_all.py's rsync_<YYYYMMDD>.log
# (RSYNC PAYLOAD / RSYNC_ACCOUNTING lines and a RSYNC TRANSFER SUMMARY).
# YYYYMMDD is the UTC start date of the run, so re-collecting a run
# appends to the same log and resumes its byte accounting.
#
# Optional daq_config field "rsync_bwlimit" (KiB/s) limits bandwidth.
#
# options when run as a cmdline script (from the control/ dir):
# --run X       run to collect (default: read from current_run)
# --node IP     only collect from this DAQ node
# --verbose

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datetime
import re
import shlex
import subprocess
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Dict

from utils import util, config_file

CATEGORIES = ("PFF", "HP_STDOUT", "PSS")


@dataclass
class TransferMeter:
    """Cumulative rsync payload transferred for one DAQ node."""
    total_bytes: int = 0
    data_date_utc: str = ""
    transfer_route: str = "DAQ -> Headnode"
    transfer_started_at: datetime.datetime = None
    transfer_finished_at: datetime.datetime = None
    category_bytes: Dict[str, int] = field(
        default_factory=lambda: {c: 0 for c in CATEGORIES}
    )


# ----------------- UTILITIES -----------------
# Console output goes through sys.stdout rather than print(),
# because stop.py replaces print() with a version that rewrites
# the daily datarec log on every call.

def log_print(message: str, logfile: str = None, end: str = "\n") -> None:
    """Print to console and optionally log to file."""
    sys.stdout.write(message + end)
    sys.stdout.flush()
    if logfile:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(message + end)


def _append_log_raw(text: str, logfile: str) -> None:
    """Append raw-ish text to logfile, keeping it readable."""
    if not logfile:
        return
    safe = text.replace("\r", "\n")
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(safe)


def run_command(cmd: str, logfile: str = None, line_handler=None) -> int:
    """
    Run a shell command and stream output live.

    - Preserves carriage returns so progress meters update cleanly on ONE line.
    - Writes a readable version to logfile (CR -> newline).
    """
    log_print(f"\n[CMD] {cmd}", logfile)

    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    pending_line = ""

    try:
        while True:
            chunk = proc.stdout.read(4096) if proc.stdout else b""
            if not chunk:
                break

            decoded = chunk.decode("utf-8", errors="replace")
            try:
                sys.stdout.write(decoded)
                sys.stdout.flush()
            except Exception:
                pass

            _append_log_raw(decoded, logfile)

            if line_handler:
                pending_line += decoded.replace("\r", "\n")
                lines = pending_line.split("\n")
                pending_line = lines.pop()
                for line in lines:
                    line_handler(line)

    except KeyboardInterrupt:
        try:
            proc.terminate()
        finally:
            raise
    finally:
        rc = proc.wait()

    if line_handler and pending_line:
        line_handler(pending_line)

    if rc == 0:
        log_print("? Done.", logfile)
    else:
        log_print(f"? Error executing: {cmd}", logfile)
        log_print(f"Command returned non-zero exit status {rc}.", logfile)

    return rc


def human_bytes(value: int) -> str:
    """Format a byte count for concise log output."""
    if value < 1024:
        return f"{value} B"
    if value < 1024 ** 2:
        return f"{value / 1024:.2f} KiB"
    if value < 1024 ** 3:
        return f"{value / 1024 ** 2:.2f} MiB"
    return f"{value / 1024 ** 3:.2f} GiB"


def bytes_from_human(value: str, unit: str) -> int:
    """Recover the approximate byte count in human-readable log entries."""
    multipliers = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}
    return int(round(float(value) * multipliers[unit]))


def utc_timestamp(value: datetime.datetime = None) -> str:
    """Return an unambiguous timestamp for the transfer log."""
    if value is None:
        value = datetime.datetime.now(datetime.timezone.utc)
    return value.astimezone(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def human_duration(duration: datetime.timedelta) -> str:
    """Format a non-negative elapsed time as HH:MM:SS."""
    total_seconds = max(0, int(duration.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_prior_transfer_totals(logfile: str) -> Dict[str, int]:
    """Load completed-transfer accounting already present in this log.

    Exact RSYNC_ACCOUNTING markers are preferred; a human-readable
    RSYNC PAYLOAD line without a following marker is used as a fallback.
    """
    totals = {c: 0 for c in CATEGORIES}
    if not os.path.isfile(logfile):
        return totals

    cats = "|".join(CATEGORIES)
    exact_pattern = re.compile(rf"^RSYNC_ACCOUNTING category=({cats}) bytes=(\d+)$")
    legacy_pattern = re.compile(
        rf"RSYNC PAYLOAD \[({cats})\]:\s*([0-9.]+)\s+(B|KiB|MiB|GiB) this transfer;"
    )
    pending_legacy = None
    with open(logfile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = exact_pattern.match(line.strip())
            if match:
                category, byte_count = match.group(1), int(match.group(2))
                # The human-readable line is written immediately before
                # this exact marker.  Replace that rounded entry rather than
                # counting the same rsync command twice.
                if pending_legacy and pending_legacy[0] != category:
                    totals[pending_legacy[0]] += pending_legacy[1]
                totals[category] += byte_count
                pending_legacy = None
                continue
            match = legacy_pattern.search(line)
            if match:
                if pending_legacy:
                    totals[pending_legacy[0]] += pending_legacy[1]
                pending_legacy = (
                    match.group(1),
                    bytes_from_human(match.group(2), match.group(3)),
                )

    if pending_legacy:
        totals[pending_legacy[0]] += pending_legacy[1]
    return totals


def run_rsync_command(
    cmd: str, logfile: str, meter: TransferMeter, category: str
) -> int:
    """Run rsync and append an updated cumulative payload total after each result."""
    stats_pattern = re.compile(r"^Total transferred file size:\s*([0-9,]+)\s+bytes$")

    def track_rsync_stats(line: str) -> None:
        match = stats_pattern.match(line.strip())
        if not match:
            return
        transferred_bytes = int(match.group(1).replace(",", ""))
        meter.total_bytes += transferred_bytes
        meter.category_bytes[category] = meter.category_bytes.get(category, 0) + transferred_bytes
        log_print(
            f"? RSYNC PAYLOAD [{category}]: {human_bytes(transferred_bytes)} this transfer; "
            f"{human_bytes(meter.total_bytes)} cumulative",
            logfile,
        )
        log_print(
            f"RSYNC_ACCOUNTING category={category} bytes={transferred_bytes}",
            logfile,
        )

    return run_command(cmd, logfile, line_handler=track_rsync_stats)


def log_transfer_summary(logfile: str, meter: TransferMeter) -> None:
    """Append an easy-to-scan per-category transfer summary to the log."""
    duration = meter.transfer_finished_at - meter.transfer_started_at
    log_print("\n=== RSYNC TRANSFER SUMMARY ===", logfile)
    log_print(f"Transfer route:                 {meter.transfer_route}", logfile)
    log_print(f"Data date transferred (UTC):    {meter.data_date_utc}", logfile)
    log_print(f"Transfer started (UTC):         {utc_timestamp(meter.transfer_started_at)}", logfile)
    log_print(f"Transfer ended (UTC):           {utc_timestamp(meter.transfer_finished_at)}", logfile)
    log_print(f"Transfer duration (HH:MM:SS):   {human_duration(duration)}", logfile)
    log_print("Data type | Transferred (all completed attempts)", logfile)
    log_print("----------+------------", logfile)
    for category in CATEGORIES:
        log_print(f"{category:<9} | {human_bytes(meter.category_bytes[category])}", logfile)
    log_print("----------+------------", logfile)
    log_print(f"GRAND TOTAL | {human_bytes(meter.total_bytes)}", logfile)


# ----------------- COLLECTION -----------------

def run_dates(run_name: str):
    """Return (YYYY-MM-DD, YYYYMMDD) of the run's UTC start time.

    Run names look like obs_X.start_2026-10-21T08:30:00Z.runtype_Y.pffd;
    fall back to the current UTC date if the name doesn't parse.
    """
    m = re.search(r"\.start_(\d{4})-(\d{2})-(\d{2})T", run_name)
    if m:
        y, mo, d = m.groups()
    else:
        now = datetime.datetime.now(datetime.timezone.utc)
        y, mo, d = now.strftime("%Y"), now.strftime("%m"), now.strftime("%d")
    return f"{y}-{mo}-{d}", f"{y}{mo}{d}"


def log_path(daq_config, node, run_name):
    _, date_nodash = run_dates(run_name)
    return '%s/%s/rsync_%s_%s.log'%(
        daq_config['head_node_data_dir'], run_name, node['ip_addr'], date_nodash
    )


def collect_node(daq_config, node, run_name, verbose=False):
    """Pull one DAQ node's files for a run into the head node run dir.

    Returns '' on success, else an error message.
    """
    run_dir = '%s/%s'%(daq_config['head_node_data_dir'], run_name)
    if not os.path.isdir(run_dir):
        return 'rsync_collect: no run dir %s'%run_dir

    date_dash, date_nodash = run_dates(run_name)
    logfile = log_path(daq_config, node, run_name)
    is_local = node['ip_addr'] in util.local_ip()

    prior = load_prior_transfer_totals(logfile)
    meter = TransferMeter(
        total_bytes=sum(prior.values()),
        data_date_utc=date_dash,
        transfer_route='DAQ %s -> Headnode'%node['ip_addr'],
        transfer_started_at=datetime.datetime.now(datetime.timezone.utc),
        category_bytes=prior,
    )

    log_print(f"\n=== ? Starting rsync transfers for UTC {date_dash} ({date_nodash}) ===", logfile)
    log_print(f"=== Run: {run_name} ===", logfile)
    log_print(f"=== Transfer route: {meter.transfer_route} ===", logfile)
    log_print(f"=== Modules: {[m['id'] for m in node['modules']]} ===\n", logfile)
    if meter.total_bytes:
        log_print(
            f"=== Resuming run log accounting: {human_bytes(meter.total_bytes)} "
            "from completed earlier attempts ===\n",
            logfile,
        )

    rsync = 'rsync -a --partial --info=progress2 --stats'
    if daq_config.get('rsync_bwlimit'):
        rsync += ' --bwlimit=%d'%daq_config['rsync_bwlimit']
    if is_local:
        remote = ''
    elif 'port_forwarding' in node:
        rsync += ' -e "ssh -p %d"'%node['port_forwarding']['port']
        remote = '%s@%s:'%(node['username'], node['port_forwarding']['gw_ip'])
    else:
        remote = '%s@%s:'%(node['username'], node['ip_addr'])

    # (category, source); sources are relative to the node's data_dir
    sources = []
    if not is_local:
        # on the head node these are already in the run dir
        # (the validator requires data_dir == head_node_data_dir)
        sources.append(('HP_STDOUT', '%s/%s*'%(run_name, util.hp_stdout_prefix)))
        sources.append(('PSS', '%s/%s*'%(run_name, util.pss_prefix)))
    for module in node['modules']:
        sources.append(('PFF', 'module_%d/%s/'%(module['id'], run_name)))

    error_msg = ''
    for category, src in sources:
        src_path = '%s/%s'%(node['data_dir'], src)
        if category == 'PFF':
            log_print(f"? uploading PFF module directory: {remote}{src_path}  ->  {run_dir}/", logfile)
            opts = ''
        else:
            log_print(f"? uploading {category}: {remote}{src_path}  ->  {run_dir}/", logfile)
            # these files may legitimately be absent
            opts = ' --ignore-missing-args'
        # a remote glob is expanded on the DAQ node, so keep it quoted;
        # a local glob must be left unquoted for the local shell
        src_arg = shlex.quote(remote + src_path)
        if not remote and src_arg.endswith("*'"):
            src_arg = src_arg[:-2] + "'*"
        cmd = '%s%s %s %s/'%(rsync, opts, src_arg, shlex.quote(run_dir))
        rc = run_rsync_command(cmd, logfile, meter, category)
        if rc:
            error_msg += 'rsync_collect: %s returned %d\n'%(cmd, rc)

    meter.transfer_finished_at = datetime.datetime.now(datetime.timezone.utc)
    log_transfer_summary(logfile, meter)
    if error_msg:
        log_print(f"\n? Transfers from {node['ip_addr']} completed with errors for {run_name}", logfile)
    else:
        log_print(f"\n? All transfers from {node['ip_addr']} completed for {run_name}", logfile)
    log_print(f"? Log saved to: {logfile}\n", logfile)
    return error_msg


# return '' if data collection was successful, else error msg
#
def collect_data(daq_config, run_name, verbose=False, node_ip=None):
    error_msg = ''
    for node in daq_config['daq_nodes']:
        if node_ip and node['ip_addr'] != node_ip:
            continue
        try:
            error_msg += collect_node(daq_config, node, run_name, verbose)
        except Exception as e:
            error_msg += 'rsync_collect: node %s failed: %s\n'%(node['ip_addr'], e)
    return error_msg


if __name__ == "__main__":
    parser = ArgumentParser(prog=os.path.basename(__file__), allow_abbrev=False)
    parser.add_argument('--run', dest='run', type=str, default=None,
                        help='Run to collect (default: current run).')
    parser.add_argument('--node', dest='node', type=str, default=None,
                        help='Only collect from the DAQ node with this IP address.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='Print details.')
    args = parser.parse_args()
    run = args.run or util.read_run_name()
    if not run:
        raise Exception("No run found")
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    network_config = config_file.get_network_config()
    util.attach_daq_config(daq_config, network_config)
    config_file.associate(daq_config, quabo_uids)
    ret = collect_data(daq_config, run, args.verbose, args.node)
    if ret:
        sys.stdout.write(ret)
    print('success' if not ret else 'failed')
    sys.exit(1 if ret else 0)
