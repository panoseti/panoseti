# control script utilities

import os, sys, subprocess, signal, socket, datetime, time, psutil, shutil
import __main__
import netifaces, json
# TODO: we need to find a better way to deal with this...
try:
    import quabo_driver
except:
    pass
import logging
#-------------- DEFAULTS ---------------

default_max_file_size_mb = 0        # no limit

#-------------- FILE NAMES ---------------

run_name_file = 'current_run'
    # stores the name of the current run
run_symlink = 'run'
    # name of symlink to current run
img_symlink = 'img'
ph_symlink= 'ph'
hk_symlink= 'hk'
    # names of symlinks to first img and ph file in current run

hk_file_name = 'hk.pff'
    # housekeeping file in run dir

# files written by stop.py
recording_ended_filename = 'recording_ended'
collect_complete_filename = 'collect_complete'
run_complete_filename = 'run_complete'

hk_recorder_name = './store_redis_data.py'

hv_updater_name = './hv_updater.py'

module_temp_monitor_name = './module_temp_monitor.py'

hashpipe_name = 'hashpipe'

daq_hashpipe_pid_filename = 'daq_hashpipe_pid'
    # stores PID of hashpipe process
daq_run_name_filename = 'daq_run_name'
    # stores name of current run
hp_stdout_prefix = 'hp_stdout_'
    # hashpipe stdout file is prefix_ipaddr
pss_prefix = 'pss_'
    # process snapshot file is pss_prefix_ipaddr
redis_daemons = [
    'capture_gps.py', 'capture_hk.py', 'capture_wr.py', 'storeInfluxDB.py'
]
#capture_power.py

#-------------- TIME ---------------

def now_str():
    t = int(time.time())
    dt = datetime.datetime.fromtimestamp(t)
    return dt.isoformat()

#-------------- NETWORK ---------------

# quabos send HK packets here at first.
# so (currently) you can only reboot quabos from this host
#
default_hk_dest = '192.168.1.100'

# create logger
#
def create_logger(logfile, tag, mode='w'):
    logger = logging.getLogger(tag)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(logfile, mode=mode)
    logformat = logging.Formatter('%(levelname)s - %(asctime)s - %(name)s - %(message)s')
    handler.setFormatter(logformat)
    if logger.handlers:
        logger.handlers.clear()
    logger.addHandler(handler)

# our IP address on local network (192.x.x.x)
# see https://pypi.org/project/netifaces/
#
def local_ip():
    ips = []
    for ifname in netifaces.interfaces():
        addrs = netifaces.ifaddresses(ifname)
        for a, b in addrs.items():
            for c in b:
                z = c['addr']
                if (z.startswith('192.')):
                    ips.append(z)
    if not ips:
        raise Exception("can't get local IP")
    else:
        return ips
    
def ip_addr_str_to_bytes(ip_addr_str):
    pieces = ip_addr_str.strip().split('.')
    if len(pieces) != 4:
        raise Exception('bad IP addr %s'%ip_addr_str)
    bytes = bytearray(4)
    for i in range(4):
        x = int(pieces[i])
        if x<0 or x>255:
            raise Exception('bad IP addr %s'%ip_addr_str)
        bytes[i] = x
    return bytes

# return true if can ping IP addr
#
def ping(ip_addr, cmd_port):
    logger = logging.getLogger('PANOSETI.Config.util.ping')
    #return not os.system('ping -c 1 -w 1 -q %s > /dev/null 2>&1'%ip_addr)
    # TODO: implement the qping cmd in the firmware
    # For now, we just use the data_packet_destination to see if we can talk to Quabo
    s = os.system('ping -c 1 -w 1 -q %s > /dev/null 2>&1'%ip_addr)
    if not s:
        return True
    else:
        quabo = quabo_driver.QUABO(ip_addr, cmd_port)
        return quabo.data_packet_destination('192.168.1.1')

def mac_addr_str(bytes):
    s = ['']*6
    for i in range(6):
        s[i] = hex(bytes[i])[2:]
    return ':'.join(s)

#-------------- BINARY DATA ---------------

def print_binary(data):
    n = len(data)
    print('got %d bytes'%n)
    for i in range(n):
        print("%d: %d"%(i, data[i]))

#-------------- QUABO OPS ---------------

# get the UID of quabo i in a given module
#
def quabo_uid(module, quabo_uids, i):
    for dome in quabo_uids['domes']:
        for m in dome['modules']:
            if m['ip_addr'] == module['ip_addr']:
                q = m['quabos'][i]
                return q['uid']
    raise Exception("no module %s found; run get_uids.py"%module['ip_addr'])

# see if quabo is alive by seeing if we got its UID
#
def is_quabo_alive(module, quabo_uids, i):
    return quabo_uid(module, quabo_uids, i) != ''

# is quabo new or old hardware version, as specified in obs_config?
# can be specified as either string or array of 4 strings
#
'''
def is_quabo_old_version(module, i):
    v = module['quabo_version']
    if isinstance(v, list):
        v = v[i]
    return v == 'qfp'
'''
def is_quabo_old_version(module, i, quabo_uids, quabo_info):
    domes = quabo_uids['domes']
    for dome in domes:
        modules = dome['modules']
        for m in modules:
            if m['ip_addr'] == module['ip_addr']:
                uid = m['quabos'][i]['uid']
    try:
        v = quabo_info[uid]['board_version']
    except:
        print('uid: %s can\'t be found in quabo_info.json'%uid)
        return
    return v == 'qfp'
#-------------- RECORDING ---------------

def start_daemon(prog):
    if is_script_running(prog):
        print('%s is already running'%prog)
        return
    try:
        process = subprocess.Popen(
            ['./'+prog], start_new_session=True,
            close_fds=True, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except:
        print("can't launch %s"%prog)
        return
    print('started %s'%prog)

# start daemons that write HK/GPS/WR data to Redis
#
def start_redis_daemons():
    for daemon in redis_daemons:
        start_daemon(daemon)

def stop_redis_daemons():
    for d in redis_daemons:
        prog = './%s'%d
        for p in psutil.process_iter():
            c = p.cmdline()
            if len(c) == 2 and c[1] == prog:
                os.kill(p.pid, signal.SIGKILL)
                print('stopped %s'%d)

def show_redis_daemons():
    for daemon in redis_daemons:
        if is_script_running(daemon):
            print('%s is running'%daemon)
        else:
            print('%s is not running'%daemon)

def are_redis_daemons_running():
    for daemon in redis_daemons:
        if not is_script_running(daemon):
            return False
    return True

def start_hk_recorder(daq_config, run_name):
    path = '%s/%s/%s'%(daq_config['head_node_data_dir'], run_name, hk_file_name)
    try:
        process = subprocess.Popen([hk_recorder_name, path])
    except:
        print("can't launch HK recorder")
        raise

        
# Start high-voltage updater daemon
def start_hv_updater():
    if is_hv_updater_running():
        print('hv_updater.py is already running')
        return
    try:
        subprocess.Popen([hv_updater_name])
    except:
        print("can't launch HV updater")
        raise


# Start module temperature monitor daemon.
def start_module_temp_monitor():
    if is_module_temp_monitor_running():
        print('module_temp_monitor.py is already running')
        return
    try:
        subprocess.Popen([module_temp_monitor_name])
    except:
        print("can't launch module temperature monitor")
        raise


# write run name to a file, and symlink 'run' to the run dir
def write_run_name(daq_config, run_name):
    with open(run_name_file, 'w') as f:
        f.write(run_name)
    if os.path.lexists(run_symlink):
        os.unlink(run_symlink)
    run_dir = '%s/%s'%(daq_config['head_node_data_dir'], run_name)
    os.symlink(run_dir, run_symlink, True)
    # record the run name in skymap_info_dir, which will be used by skymap_helper
    os.system('cp %s %s'%(run_name_file, 'skymap_info_dir'))

def read_run_name():
    if not os.path.exists(run_name_file):
        return None
    with open(run_name_file) as f:
        return f.read()

def remove_run_name():
    if os.path.exists(run_name_file):
        os.unlink(run_name_file)

# if hashpipe is running, send it a SIGINT and wait for it to exit
#
def stop_hashpipe(pid):
    for p in psutil.process_iter():
        if p.pid == pid and p.name() == hashpipe_name:
            os.kill(pid, signal.SIGINT)
            while True:
                try:
                    os.kill(pid, 0)
                except:
                    return True
                time.sleep(0.1)
    return False

def is_script_running(script):
    s = './%s'%script
    for p in psutil.process_iter():
        if s in p.cmdline():
            return True
    return False

def is_hashpipe_running():
    for p in psutil.process_iter():
        if p.name() == hashpipe_name:
            return True;
    return False

def is_hk_recorder_running():
    for p in psutil.process_iter():
        if hk_recorder_name in p.cmdline():
            return True
    return False

def is_hv_updater_running():
    return is_script_running(hv_updater_name[2:])

def is_module_temp_monitor_running():
    return is_script_running(hv_updater_name[2:])

def kill_hashpipe():
    for p in psutil.process_iter():
        if p.name() == hashpipe_name:
            os.kill(p.pid, signal.SIGKILL)

def kill_hk_recorder():
    for p in psutil.process_iter():
        if hk_recorder_name in p.cmdline():
            os.kill(p.pid, signal.SIGKILL)

def kill_hv_updater():
    for p in psutil.process_iter():
        if hv_updater_name in p.cmdline():
            os.kill(p.pid, signal.SIGKILL)


def kill_module_temp_monitor():
    for p in psutil.process_iter():
        if module_temp_monitor_name in p.cmdline():
            os.kill(p.pid, signal.SIGKILL)

# write a message to per-run log file, and to stdout
#
def write_log(msg):
    now = datetime.datetime.now().strftime("%B %d, %Y, %I:%M%p")
    print('%s: %s: %s'%(__main__.__file__, now, msg))
    try:
        f = open('run/log.txt', 'a')
        f.write('%s: %s: %s'%(__main__.__file__, now, msg))
        f.close()
    except:
        f = open('log.txt', 'a')

def disk_usage(dir):
    x = 0
    for f in os.listdir(dir):
        x += os.path.getsize('%s/%s'%(dir, f))
    return x

def free_space(path):
    total, used, free = shutil.disk_usage(os.path.realpath(path))
    return free

# estimate bytes per second per module for a given data config
def daq_bytes_per_sec_per_module(data_config):
    img_json_header_size = 600
    ph_json_header_size = 150
    x = 0

    # hk.pff
    x += 2000 + 800*4

    if 'image' in data_config:
        image = data_config['image']
        fps = 1e6/image['integration_time_usec']
        if image['quabo_sample_size'] == 8:
            bpf = 1
        else:
            bpf = 2
        x += fps*(1024*bpf + img_json_header_size)
    if 'pulse_height' in data_config:
        # assume one PH event per sec per quabo
        ph_per_sec = 1
        x += ph_per_sec*(4*(256*2+ph_json_header_size))
    return x

def get_daq_node_status(node):
    # TODO: add port forwarding code here
    x = subprocess.run(['ssh',
        '%s@%s'%(node['username'], node['ip_addr']),
        'cd %s; ./status_daq.py'%(node['data_dir']),
        ],
        stdout = subprocess.PIPE
    )
    if x=='':
        raise Exception("can't talk to DAQ node")
    y = x.stdout.decode()
    return json.loads(y)

#-------------- functions only for DAQ nodes ---------------

def daq_get_run_name():
    if os.path.exists(daq_run_name_filename):
        with open(daq_run_name_filename) as f:
            return f.read().strip()
    return None


#-------------- WR and GPS---------------

def get_wr_ip_addr(obs_config):
    if 'wr_ip_addr' in obs_config.keys():
        return obs_config['wr_ip_addr']
    else:
        return '192.168.1.254'

# get GPS receiver port (path of the tty)
#
def get_gps_port(obs_config):
    if 'gps_port' in obs_config.keys():
        return obs_config['gps_port']
    else:
        return '/dev/ttyUSB0'

# We may use port forwarding, so we need to get the real IP and ports.
# this is based on the network_config.
#
DEFAULT_CMD_PORT=60000
DEFAULT_REBOOT_PORT=69
def get_quabo_ip_port(ip_addr, i, network_config):
    ip_ports = {}
    x = ip_addr.split('.')
    x[3] = str(int(x[3])+i)
    quabo_ip =  '.'.join(x)
    # these are the default config
    ip_ports['ip_addr'] = quabo_ip
    ip_ports['reboot_port'] = DEFAULT_REBOOT_PORT
    ip_ports['cmd_port'] = DEFAULT_CMD_PORT
    # if we can't find the setting for the Quabo in the network_config
    # we will use the default config
    for m in network_config['modules']:
        if ip_addr == m['ip_addr']:
            p = m['port_forwarding']
            if p['status'] == True:
                ip_ports['ip_addr'] = p['gw_ip']
                ip_ports['reboot_port'] = p['reboot_port'][i]
                ip_ports['cmd_port'] = p['cmd_port'][i]
            break
    return ip_ports

# attach port forwarding info to daq config based on network_config        
def attach_daq_config(daq_config, network_config):
    for i in range(len(daq_config['daq_nodes'])):
        daq = daq_config['daq_nodes'][i]
        for pdaq in network_config['daq_nodes']:
             if daq['ip_addr'] == pdaq['ip_addr'] and pdaq['port_forwarding']['status'] == True:
                 daq_config['daq_nodes'][i]['port_forwarding'] = pdaq['port_forwarding']