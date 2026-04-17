#! /usr/bin/env python3

# stop and finish a recording run if one is in progress.
# stop recording activities whether or not a run is in progress.
#
# - tell DAQs to stop recording
# - stop HK recorder process
# - tell quabos to stop sending data
# - if a run is in progress, copy data files to head and delete from DAQs
#
# options:
#   --verbose           print details
#   --no_collect        don't copy data files to head node
#   --no_cleanup        don't delete files from DAQ nodes
#   --run X             clean up run X (default: read from current_run)

import builtins
import contextlib
import logging
import os
import signal
import socket
import sys
import tempfile
import time
from argparse import ArgumentParser
from datetime import UTC, datetime
from typing import Any

from panoseti_grpc.daq_control.client import DaqControlClient

from driver import quabo_driver
from tools.interleave import PID_FILE
from utils import collect, config_file, pff
from utils.util import (
    attach_daq_config,
    collect_complete_filename,
    create_logger,
    daq_grpc_endpoint,
    get_quabo_ip_port,
    hk_symlink,
    img_symlink,
    kill_hk_recorder,
    kill_hv_updater,
    kill_module_temp_monitor,
    local_ip,
    now_str,
    ph_symlink,
    read_run_name,
    recording_ended_filename,
    remove_run_name,
    run_complete_filename,
)

logger = logging.getLogger(__name__)


def stop_interleave(retry_limit: int = 10) -> None:
    """
    Checks if the interleave process is running and cleanly shuts it down.
    This prevents background mode switching after DAQ has been commanded to stop.
    """
    pid_file = PID_FILE
    if os.path.exists(pid_file):
        print("Active interleave process detected. Stopping it gracefully...")
        try:
            with open(pid_file) as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)

            # Wait briefly for it to clean up and restore defaults

            for r in range(retry_limit):
                if not os.path.exists(pid_file):
                    break
                logger.warning(f"Stopping interleave process: {pid}. Attempt [{r}/{retry_limit}]")
                time.sleep(0.5)
        except (OSError, ValueError):
            os.remove(pid_file)


# =========================
# Print -> also prepend to UT log file
# =========================

_ORIG_PRINT = builtins.print

def _ut_human_timestamp() -> str:
    # Human-readable UTC timestamp
    return datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UT')

def _ut_yyyymmdd() -> str:
    return datetime.now(UTC).strftime('%Y%m%d')

def _datarec_log_path() -> str:
    yyyymmdd = _ut_yyyymmdd()
    log_dir = f"/mnt/data11/data/palomar/L0/{yyyymmdd}/obslogs"
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"datarec_{yyyymmdd}.log")

def _prepend_line_to_file(path: str, line: str) -> None:
    # Prepend efficiently by writing a temp file then replacing.
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

    old = b""
    try:
        with open(path, "rb") as f:
            old = f.read()
    except FileNotFoundError:
        old = b""

    new_bytes = (line + "\n").encode("utf-8", errors="replace")

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_datarec_", dir=d or None)
    try:
        with os.fdopen(fd, "wb") as tf:
            tf.write(new_bytes)
            tf.write(old)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass

def print(*args: Any, **kwargs: Any) -> None:
    # Console print as-is + prepend to UT log file with timestamp.
    msg = " ".join(str(a) for a in args)
    with contextlib.suppress(Exception):
        _prepend_line_to_file(_datarec_log_path(), f"{_ut_human_timestamp()}: {msg}")
    _ORIG_PRINT(*args, **kwargs)

builtins.print = print

# write message to error log
#
def log_error(msg: str, run_dir: str | None) -> None:
    print(msg)
    log_path = f'{run_dir}/stop_errors' if run_dir else 'stop_errors'
    with open(log_path, 'a') as f:
        f.write(f'{now_str()}: {msg}\n')

# tell the quabos to stop sending data
#
def stop_data_flow(quabo_uids: dict[str, Any], network_config: dict[str, Any]) -> None:
    logger = logging.getLogger('PANOSETI.Stop.stop_data_flow')
    daq_params = quabo_driver.DAQ_PARAMS(False, 0, False, False, False)
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            if 'daq_node' not in module:
                continue
            base_ip_addr = module['ip_addr']
            for i in range(4):
                quabo_uid = module['quabos'][i]
                if quabo_uid['uid'] == '':
                    continue
                ip_addr = config_file.quabo_ip_addr(base_ip_addr, i)
                ip_ports = get_quabo_ip_port(base_ip_addr, i, network_config)
                real_ip = ip_ports['ip_addr']
                cmd_port = ip_ports['cmd_port']
                logger.info(f'Quabo IP: {ip_addr}')
                logger.info(f'Real IP: {real_ip}')
                logger.info(f'Cmd Port: {cmd_port}')
                quabo = quabo_driver.QUABO(real_ip, cmd_port)
                quabo.send_daq_params(daq_params)
                quabo.close()

# tell all DAQ nodes to stop recording
#
def stop_recording(daq_config: dict[str, Any], run_dir: str | None, verbose: bool) -> None:
    logger = logging.getLogger('PANOSETI.Stop.stop_recording')
    for node in daq_config['daq_nodes']:
        grpc_host, grpc_port = daq_grpc_endpoint(node)
        if verbose:
            print(f'StopDaq via gRPC: {grpc_host}:{grpc_port}')
        logger.info(f'StopDaq via gRPC: {grpc_host}:{grpc_port}')
        client = DaqControlClient(host=grpc_host, port=grpc_port)
        ok = client.StopDaq({
            'data_dir': node['data_dir'],
            'run_dir':  run_dir,
        })
        if not ok:
            msg = f'StopDaq failed for node {node["ip_addr"]}'
            logger.error(msg)
            raise Exception(msg)

# write a "complete file" in the run dir
#
def write_complete_file(run_dir: str, filename: str) -> None:
    path = f'{run_dir}/{filename}'
    with open(path , 'w') as f:
        f.write(now_str())

def complete_file_exists(run_dir: str, filename: str) -> bool:
    path = f'{run_dir}/{filename}'
    return os.path.exists(path)

# make symlinks to the first nonempty image and ph files in that dir
#
def make_links(run_dir: str, verbose: bool) -> None:
    if os.path.lexists(img_symlink):
        os.unlink(img_symlink)
    if os.path.lexists(ph_symlink):
        os.unlink(ph_symlink)
    if os.path.lexists(hk_symlink):
        os.unlink(hk_symlink)
    did_img = False
    did_ph = False
    did_hk = False
    for f in os.listdir(run_dir):
        path = f'{run_dir}/{f}'
        if not pff.is_pff_file(path):
            continue
        if os.path.getsize(path) == 0:
            continue
        ftype = pff.pff_file_type(f)
        if not did_img and ftype in ['img16', 'img8']:
            os.symlink(path, img_symlink)
            did_img = True
            if verbose:
                print(f'linked {img_symlink} to {f}')
        elif not did_ph and ftype in ['ph256', 'ph1024']:
            os.symlink(path, ph_symlink)
            did_ph = True
            if verbose:
                print(f'linked {ph_symlink} to {f}')
        elif not did_hk and ftype == 'hk':
            os.symlink(path, hk_symlink)
            did_hk = True
            if verbose:
                print(f'linked {hk_symlink} to {f}')
        if did_img and did_ph and did_hk:
            break
    if not did_img:
        print('make_links(): No nonempty image file')
    if not did_ph:
        print('make_links(): No nonempty PH file')
    if not did_hk:
        print('make_links(): No nonempty housekeeping file')


def _cleanup_daq_grpc(daq_config: dict[str, Any], run: str, head_run_dir: str, verbose: bool) -> None:
    """Call CleanupData on each DAQ node via gRPC.
    Only called after collect_data() succeeds (transactional guarantee).
    CleanupData is blocked server-side if hashpipe is still running.
    """
    logger = logging.getLogger('PANOSETI.Stop._cleanup_daq_grpc')
    my_ip = local_ip()
    for node in daq_config['daq_nodes']:
        if node['ip_addr'] in my_ip:
            # Head node is also DAQ node: local rm -rf (same as before)
            cmd = 'rm -rf {}/module_*/{}'.format(node['data_dir'], run)
            if verbose:
                print(cmd)
            ret = os.system(cmd)
            if ret:
                log_error(f'cleanup_daq (local): {cmd} returned {ret}', head_run_dir)
        else:
            module_ids = [m['id'] for m in node.get('modules', [])]
            grpc_host, grpc_port = daq_grpc_endpoint(node)
            if verbose:
                print(f'CleanupData via gRPC: {grpc_host}:{grpc_port}  run_dir={run}  modules={module_ids}')
            logger.info(f'CleanupData via gRPC: {grpc_host}:{grpc_port}')
            try:
                client = DaqControlClient(host=grpc_host, port=grpc_port)
                cleanup_resp = client.CleanupData({
                    'data_dir':  node['data_dir'],
                    'run_dir':   run,
                    'module_id': module_ids,
                })
                if not cleanup_resp['success']:
                    log_error(f'CleanupData failed for node {node["ip_addr"]} with {cleanup_resp=}', head_run_dir) 
            except Exception as e:
                log_error(f'CleanupData error for node {node["ip_addr"]}: {e}', head_run_dir)


def stop_run(
    daq_config: dict[str, Any], network_config: dict[str, Any], quabo_uids: dict[str, Any], verbose: bool = False, no_cleanup: bool = False, no_collect: bool = False,
    run: str | None = None
) -> None:
    # convert head node name to IP address
    head_node_ip = socket.gethostbyname(daq_config['head_node_ip_addr'])
    if head_node_ip not in local_ip():
        raise Exception(
            'This computer ({}) is not the head node specified in daq_config.json ({})'.format(
                local_ip(), daq_config['head_node_ip_addr']
            )
        )

    if not run:
        run = read_run_name()
    if not run:
        print("No run is in progress")
        return

    data_dir = daq_config['head_node_data_dir']
    run_dir: str | None = f'{data_dir}/{run}'
    if run_dir is not None and not os.path.exists(run_dir):
        run_dir = None

    # do things that don't depend on having a run dir

    print("stopping data recording")
    stop_recording(daq_config, run, verbose)

    print("stopping HV updater")
    kill_hv_updater()

    print("stopping HK recording")
    kill_hk_recorder()

    print("stopping Temperature monitor")
    kill_module_temp_monitor()

    print("stopping data generation")
    stop_data_flow(quabo_uids, network_config)

    if run_dir:
        if not complete_file_exists(run_dir, recording_ended_filename):
            write_complete_file(run_dir, recording_ended_filename)
        collect_error = ''
        if not no_collect and not complete_file_exists(run_dir, collect_complete_filename):
            print("collecting data from DAQ nodes")
            collect_error = collect.collect_data(daq_config, run, verbose)
            if collect_error == '':
                write_complete_file(run_dir, collect_complete_filename)
        if collect_error == '':
            if not no_cleanup:
                if verbose:
                    print("cleaning up DAQ nodes via gRPC CleanupData")
                _cleanup_daq_grpc(daq_config, run, run_dir, verbose)
            make_links(run_dir, verbose)
            write_complete_file(run_dir, run_complete_filename)
            print(f'completed run {run}')
        else:
            log_error(collect_error, run_dir)
        remove_run_name()
    else:
        print(f"Run dir {data_dir}/{run} not found; recorded artifacts may be missing.")

if __name__ == "__main__":
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logfile = 'logs/stop.log'
    create_logger(logfile, 'PANOSETI.Stop', 'a')
    logger = logging.getLogger('PANOSETI.Stop')
    logger.info('************************************')
    i = 1
    argv = sys.argv
    verbose = False
    no_cleanup = False
    no_collect = False
    run = None
    parser = ArgumentParser(prog=os.path.basename(__file__), allow_abbrev=False)
    parser.add_argument('--no_cleanup', dest='no_cleanup', action='store_true', default=False,
                        help='Don\'t clean up the data files on the DAQ nodes.')
    parser.add_argument('--no_collect', dest='no_collect', action='store_true', default=False,
                        help='Don\'t collect the data files to the head node.')
    parser.add_argument('--run', dest='run', type=str, default=None,
                        help='Move the data files for the specific run to the head node.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='Print commands.')
    args = parser.parse_args()
    verbose = args.verbose
    no_cleanup = args.no_cleanup
    no_collect = args.no_collect
    run = args.run
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    network_config = config_file.get_network_config()
    attach_daq_config(daq_config, network_config)
    config_file.associate(daq_config, quabo_uids)

    # Kill interleaving before stopping primary data flow
    try:
        stop_interleave(retry_limit=10)
    except Exception as e:
        logger.critical('Failed to stop interleave!')
        logger.exception(e)

    # Stop run
    stop_run(daq_config, network_config, quabo_uids, verbose, no_cleanup, no_collect, run)


