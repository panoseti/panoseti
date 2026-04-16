#! /usr/bin/env python3

# start.py [--no_hv] [--no_redis] [--no_data] [--verbose] [--help]
#          [--nsecs N] [--stop_session]
#
# start a recording run:
#
# - figure out association of quabos and DAQ nodes,
#   based on config files
# - create "run directories" on head node, DAQ nodes
# - start the HK recorder
# - start the HV updater
# - start the temperature monitor
# - start the flow of data: set DAQ mode and dest IP addr of quabos
# - send commands to DAQ nodes to start hashpipe program
#
# fail if a recording run is in progress,
# or if recording activities are active
#

# based on matlab/startmodules.m, startqNph.m, changepeq.m

# ---------------- PRINT -> UT TIMESTAMP + FILE LOG ----------------
import builtins
import logging
import os
import shutil
import socket
import time
import traceback
from argparse import ArgumentParser
from datetime import UTC, datetime
from glob import glob
from typing import Any

from panoseti_grpc.daq_control.client import DaqControlClient

import session_stop
import stop
from driver import quabo_driver
from tools.sw_info import get_sw_info
from utils import config_file, file_xfer, pff, util

_LOG_ROOT = "/mnt/data11/data/palomar/L0"

def _ut_yyyymmdd() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")

def _ut_human_ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UT")

def _datarec_log_path() -> tuple[str, str]:
    yyyymmdd = _ut_yyyymmdd()
    obslogs_dir = os.path.join(_LOG_ROOT, yyyymmdd, "obslogs")
    return obslogs_dir, os.path.join(obslogs_dir, f"datarec_{yyyymmdd}.log")

_orig_print = builtins.print

def _print(*args: Any, **kwargs: Any) -> None:
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    file_arg = kwargs.get("file")
    flush = kwargs.get("flush", False)

    msg = sep.join(str(a) for a in args)
    line = f"{_ut_human_ts()} {msg}"

    # Console (or provided file), with timestamp prepended
    _orig_print(line, sep=sep, end=end, file=file_arg, flush=flush)

    # Append at the beginning of daily log file (best-effort; no extra prints)
    try:
        obslogs_dir, log_path = _datarec_log_path()
        os.makedirs(obslogs_dir, exist_ok=True)

        # Read existing contents (if any), then write new line + old contents
        try:
            with open(log_path, encoding="utf-8") as f:
                old = f.read()
        except FileNotFoundError:
            old = ""

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(line + end)
            if old:
                f.write(old)
            if flush and hasattr(f, "flush"):
                f.flush()
    except Exception:
        pass

builtins.print = _print
# ------------------------------------------------------------------

verbose = False

# check that PH calibration file is present, nonempty, and at most 24 hours old
#
def ph_baseline_file_ok() -> bool:
    if not os.path.exists(config_file.quabo_ph_baseline_filename):
        print('quabo_ph_baseline.json not found.  Run config.py --calibrate_ph')
        return False
    if os.path.getmtime(config_file.quabo_ph_baseline_filename) < time.time() - 24*86400:
        print('quabo_ph_baseline.json is too old.  Run config.py --calibrate_ph')
        return False
    return True


# check validity of image params (rate, bpp)
#
def check_img_params(image_8bit: bool, image_usec: int) -> None:
    if image_8bit:
        if image_usec < 20 or image_usec > 25:
            raise Exception('integration time must be 20-25 usec in 8 bit mode')
    else:
        if image_usec < 40:
            raise Exception('integration time must be >= 40 usec in 16 bit mode')

# parse the data config file to get DAQ params for quabos
#
def get_daq_params(data_config: dict[str, Any]) -> quabo_driver.DAQ_PARAMS:
    do_image = False
    image_usec = 1
    image_8bit = False
    do_ph = False
    bl_subtract = True
    do_any_trigger = False
    group_ph_frames = False
    if 'image' in data_config:
        do_image = True
        image = data_config['image']
        if 'integration_time_usec' not in image:
            raise Exception('missing integration_time_usec in data_config.json')
        if image['quabo_sample_size'] == 8:
            image_8bit = True
        elif image['quabo_sample_size'] != 16:
            raise Exception('quabo_sample_size must be 8 or 16')
        image_usec = image['integration_time_usec']
        #check_img_params(image_8bit, image_usec)
    if 'pulse_height' in data_config:
        do_ph = True
        if 'any_trigger' in data_config['pulse_height']:
            do_any_trigger = True
            any_trigger = data_config['pulse_height']['any_trigger']
            if 'group_ph_frames' not in any_trigger:
                raise Exception('missing "group_ph_frames" param for "any_trigger" in data_config.json')
            if any_trigger['group_ph_frames'] == 1:
                group_ph_frames = True
            elif any_trigger['group_ph_frames'] != 0:
                raise Exception('group_ph_frames for any_trigger in data_config.json must be 0 or 1.')
    daq_params = quabo_driver.DAQ_PARAMS(
        do_image, image_usec - 1, image_8bit, do_ph, bl_subtract, do_any_trigger, group_ph_frames
    )
    if 'flash_params' in data_config:
        fp = data_config['flash_params']
        daq_params.set_flash_params(fp['rate'], fp['level'], fp['width'])
    if 'stim_params' in data_config:
        sp = data_config['stim_params']
        daq_params.set_stim_params(sp['rate'], sp['level'])
    return daq_params

# Start data flow from the quabos.
# For each quabo:
# - tell it where to send HK packets
# - tell it where to send data packets
# - set its DAQ mode
#
def start_data_flow(quabo_uids: dict[str, Any], data_config: dict[str, Any], daq_config: dict[str, Any], network_config: dict[str, Any]) -> None:
    logger = logging.getLogger('PANOSETI.Start.start_data_flow')
    daq_params = get_daq_params(data_config)        
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            if 'daq_node' not in module:
                continue
            base_ip_addr = module['ip_addr']
            module_id = config_file.ip_addr_to_module_id(base_ip_addr)
            daq_node = config_file.module_id_to_daq_node(daq_config, module_id)
            daq_node_ip_addr = daq_node['ip_addr']
            head_node_ip_addr = daq_config['head_node_ip_addr']
            for i in range(4):
                quabo_uid = module['quabos'][i]
                if quabo_uid['uid'] == '':
                    continue
                ip_addr = config_file.quabo_ip_addr(base_ip_addr, i)
                ip_ports = util.get_quabo_ip_port(base_ip_addr, i, network_config)
                real_ip = ip_ports['ip_addr']
                cmd_port = ip_ports['cmd_port']
                logger.info(f'Quabo IP: {ip_addr}')
                logger.info(f'Real IP: {real_ip}')
                logger.info(f'Cmd Port: {cmd_port}')
                quabo = quabo_driver.QUABO(real_ip, cmd_port)
                if verbose:
                    print(f'setting HK packet dest to {head_node_ip_addr} on quabo {ip_addr}')
                quabo.hk_packet_destination(head_node_ip_addr)
                if verbose:
                    print(f'setting data packet dest to {daq_node_ip_addr} on quabo {ip_addr}')
                quabo.data_packet_destination(daq_node_ip_addr)
                if verbose:
                    print(f'setting DAQ mode on quabo {ip_addr}')
                quabo.send_daq_params(daq_params)
                quabo.close()
            # send software 1PPS
            time.sleep(0.5)
            logger.info(f'Send software 1PPS to {base_ip_addr}')
            ip_ports = util.get_quabo_ip_port(base_ip_addr, 0, network_config)
            real_ip = ip_ports['ip_addr']
            cmd_port = ip_ports['cmd_port']
            quabo = quabo_driver.QUABO(real_ip, cmd_port)
            quabo.swpps()
            quabo.close()

# make run directories; copy config files to them
# on each DAQ node:
# data/
#     run/         config files go here
#     module_n
#         run/     .pff files go here
#
def make_run_dirs(run_name: str, daq_config: dict[str, Any]) -> None:
    logger = logging.getLogger('PANOSETI.Start')
    my_ip = util.local_ip()
    run_dir = '{}/{}'.format(daq_config['head_node_data_dir'], run_name)
    os.mkdir(run_dir)

    # copy config files to run dir on this node
    daq_config['head_node_data_dir']
    for f in config_file.config_file_names:
        files = glob(f)
        for file in files:
            fparts = file.split('/')
            shutil.copyfile(file, f'{run_dir}/{fparts[-1]}')
     
    # make module and run directories on DAQ nodes
    #
    for node in daq_config['daq_nodes']:
        if not node['modules']:
            continue
        ip_addr = node['ip_addr']
        if ip_addr in my_ip:
            for module in node['modules']:
                cmd = f'mkdir -p {daq_config["head_node_data_dir"]}/module_{module["id"]}/{run_name}'
                if verbose:
                    print(cmd)
                ret = os.system(cmd)
                if ret:
                    raise Exception(f'{cmd} returned {ret}')
        else:
            username = node['username']
            data_dir = node['data_dir']
            rcmds = [f'mkdir {data_dir}/{run_name}']
            for module in node['modules']:
                rcmds.append(f'mkdir -p {data_dir}/module_{module["id"]}/{run_name}')
            # create process snapshot
            rcmds.append(f'cd {data_dir}/{run_name}; ps -ux > pss_{ip_addr}.log')
            rcmd = ';'.join(rcmds)
            logger.info(f'DAQ IP: {ip_addr}')
            if 'port_forwarding' in node:
                real_ip = node['port_forwarding']['gw_ip']
                port = node['port_forwarding']['port']
                logger.info('Use port forwarding')
                logger.info(f'Real IP: {real_ip}')
                logger.info(f'Port: {port}')
                cmd = f'ssh -p {port} {username}@{real_ip} "{rcmd}"'
            else:
                cmd = f'ssh {username}@{ip_addr} "{rcmd}"'
            if verbose:
                print(cmd)
            ret = os.system(cmd)
            if ret:
                raise Exception(f'{cmd} returned {ret}')

    # copy config files to DAQ nodes
    file_xfer.copy_config_files(daq_config, run_name, verbose)

# start recording data
#   for each DAQ node that is getting data
#       create run directory
#       copy config files to run directory
#       start hashpipe program
#
def start_recording(obs_config: dict[str, Any], data_config: dict[str, Any], daq_config: dict[str, Any], run_name: str, no_hv: bool) -> None:
    logger = logging.getLogger('PANOSETI.Start.start_recording')
    util.local_ip()

    # start recording HK data
    util.start_hk_recorder(daq_config, run_name)
    obs = obs_config['name']
    if not no_hv:
        # start high-voltage updater
        util.start_hv_updater()

        # start module temperature monitor
        util.start_module_temp_monitor()

    # start hashpipe on DAQ nodes

    if 'max_file_size_mb' in data_config.keys():
        max_file_size_mb = int(data_config['max_file_size_mb'])
    else:
        max_file_size_mb = util.default_max_file_size_mb
    daq_params = get_daq_params(data_config)
    for node in daq_config['daq_nodes']:
        if not node['modules']:
            continue
        data_dir = node['data_dir']
        module_ids = [
            config_file.ip_addr_to_module_id(m['ip_addr']) for m in node['modules']
        ]
        grpc_host, grpc_port = util.daq_grpc_endpoint(node)
        if verbose:
            print(f'StartDaq via gRPC: {grpc_host}:{grpc_port}  modules={module_ids}')
        logger.info(f'StartDaq via gRPC: {grpc_host}:{grpc_port}  modules={module_ids}')
        client = DaqControlClient(host=grpc_host, port=grpc_port)
        ok = client.StartDaq({
            'data_dir':         data_dir,
            'daq_ip_addr':      str(node['ip_addr']),
            'bindhost':         node.get('bindhost', '0.0.0.0'),
            'max_file_size_mb': int(max_file_size_mb),
            'group_ph_frames':  bool(daq_params.do_group_ph_frames),
            'run_dir':          run_name,
            'obs':              obs,
            'module_id':        module_ids,
        })
        if not ok:
            raise Exception(f'StartDaq failed for node {node["ip_addr"]}')

def start_run(
    obs_config: dict[str, Any], daq_config: dict[str, Any], quabo_uids: dict[str, Any], data_config: dict[str, Any], no_hv: bool, no_redis: bool, no_data: bool
) -> bool:
    my_ip = util.local_ip()
    # convert head node name to IP address
    head_node_ip = socket.gethostbyname(daq_config['head_node_ip_addr'])
    if  head_node_ip not in my_ip:
        print('This node ({}) is not the head node specified in daq_config.json ({})'.format(my_ip, daq_config['head_node_ip_addr']))
        return False

    rn = util.read_run_name()
    if (rn):
        print(f'A run is already in progress: {rn}')
        print('Run stop.py, then try again.')
        return False

    if util.is_hk_recorder_running():
        print('The HK recorder is running.  Run stop.py, then try again.')
        return False
        
    if not no_redis:
        if not util.are_redis_daemons_running():
            print('Redis daemons are not running.  Run config.py --redis_daemons')
            util.show_redis_daemons()
            return False

    if not ph_baseline_file_ok():
        return False

    # get git commit info, and write the info into sw_info.json
    get_sw_info()
    # if head node is also DAQ node, make sure data dirs are the same
    #
    for node in daq_config['daq_nodes']:
        if my_ip == node['ip_addr'] and daq_config['head_node_data_dir'] != node['data_dir']:
            print("Head node data dir doesn't match DAQ node data dir")
            return False

    try:
        run_name = pff.run_dir_name(obs_config['name'], data_config['run_type'])
        config_file.associate(daq_config, quabo_uids)
        config_file.show_daq_assignments(quabo_uids)
        print('setting up run directories')
        make_run_dirs(run_name, daq_config)
        if not no_data:
            print('starting data flow from quabos')
            start_data_flow(quabo_uids, data_config, daq_config, network_config)
            print('starting recording')
            start_recording(obs_config, data_config, daq_config, run_name, no_hv)
    except Exception:
        print(traceback.format_exc())
        print("Couldn't start run.  Run stop.py, then try again.")
        print('If other users might be using the telescope, check with them;')
        print('running stop.py will kill their run.')
        return False
    util.write_run_name(daq_config, run_name)
    print(f'started run {run_name}')
    return True

if __name__ == "__main__":
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logfile = 'logs/start.log'
    util.create_logger(logfile, 'PANOSETI.Start', 'a')
    logger = logging.getLogger('PANOSETI.Start')
    logger.info('************************************')
    parser = ArgumentParser(prog=os.path.basename(__file__), allow_abbrev=False)
    parser.add_argument('--no_hv', dest='no_hv', action='store_true', default=False,
                        help='Take data without high voltage.')
    parser.add_argument('--no_redis', dest='no_redis', action='store_true', default=False,
                        help='OK if redis daemons not running.')
    parser.add_argument('--no_data', dest='no_data', action='store_true', default=False,
                        help='Set up to record, but don\'t start data flow or record.')
    parser.add_argument('--nsecs', dest='nsecs', type=int, default=0,
                        help='Record for N seconds, then stop run.')
    parser.add_argument('--stop_session', dest='stop_session', action='store_true', default=False,
                        help='Stop session at end of run (with --nsecs).')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='print commands.')
    args = parser.parse_args()
    no_hv = args.no_hv
    no_redis = args.no_redis
    no_data = args.no_data
    nsecs = args.nsecs
    stop_session = args.stop_session
    verbose = args.verbose
    # load config files
    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    network_config = config_file.get_network_config()
    util.attach_daq_config(daq_config, network_config)
    start_run(
        obs_config, daq_config, quabo_uids, data_config,
        no_hv, no_redis, no_data
    )
    if nsecs:
        time.sleep(nsecs)
        stop.stop_run(daq_config, network_config, quabo_uids)
        if stop_session:
            session_stop.session_stop(obs_config)
