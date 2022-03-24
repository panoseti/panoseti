# control script utilities

import os, sys, subprocess, signal, socket, datetime, time, psutil, shutil

#-------------- DEFAULTS ---------------

default_max_file_size_mb = 1000

#-------------- FILE NAMES ---------------

run_name_file = 'current_run'
    # stores the name of the current run

hk_file_name = 'hk.pff'
    # housekeeping file in run dir

run_complete_file = 'run_complete'

hk_recorder_name = './store_redis_data.py'

hashpipe_name = 'hashpipe'

daq_hashpipe_pid_filename = 'daq_hashpipe_pid'
    # stores PID of hashpipe process
daq_run_name_filename = 'daq_run_name'
    # stores name of current run
hp_stdout_prefix = 'hp_stdout_'
    # hashpipe stdout file is prefix_ipaddr

redis_daemons = ['capture_gps.py', 'capture_hk.py', 'capture_wr.py']

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

def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    x = s.getsockname()[0]
    s.close()
    return x

def ip_addr_str_to_bytes(ip_addr_str):
    pieces = ip_addr_str.split('.')
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
def ping(ip_addr):
    return not os.system('ping -c 1 -w 1 -q %s > /dev/null 2>&1'%ip_addr)

# compute a 'module ID', given its base quabo IP addr: bits 2..9 of IP addr
#
def ip_addr_to_module_id(ip_addr_str):
    pieces = ip_addr_str.split('.')
    n = int(pieces[3]) + 256*int(pieces[2])
    return (n>>2)&255

#-------------- BINARY DATA ---------------

def print_binary(data):
    n = len(data)
    print('got %d bytes'%n)
    for i in range(n):
        print("%d: %d"%(i, data[i]))

#-------------- QUABO OPS ---------------

# given module base IP address, return IP addr of quabo i
#
def quabo_ip_addr(base, i):
    x = base.split('.')
    x[3] = str(int(x[3])+i)
    return '.'.join(x)

# see if quabo is alive by seeing if we got its UID
#
def is_quabo_alive(module, quabo_uids, i):
    for dome in quabo_uids['domes']:
        for m in dome['modules']:
            if m['ip_addr'] == module['ip_addr']:
                q = m['quabos'][i]
                return q['uid'] != ''
    raise Exception("no such module")

# is quabo new or old hardware version, as specified in obs_config?
# can be specified as either string or array of 4 strings
#
def is_quabo_old_version(module, i):
    v = module['quabo_version']
    if isinstance(v, list):
        v = v[i]
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

def write_run_name(run_name):
    with open(run_name_file, 'w') as f:
        f.write(run_name)

def read_run_name():
    if not os.path.exists(run_name_file):
        return None
    with open(run_name_file) as f:
        return f.read()

def remove_run_name():
    if os.path.exists(run_name_file):
        os.unlink(run_name_file)

def write_run_complete_file(daq_config, run_name):
    path = '%s/%s/%s'%(daq_config['head_node_data_dir'], run_name, run_complete_file)
    with open(path, 'w') as f:
        f.write(now_str())

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

def kill_hashpipe():
    for p in psutil.process_iter():
        if p.name() == hashpipe_name:
            os.kill(p.pid, signal.SIGKILL)

def kill_hk_recorder():
    for p in psutil.process_iter():
        if (hk_recorder_name in p.cmdline()) and p.username() == os.getlogin():
            os.kill(p.pid, signal.SIGKILL)

def disk_usage(dir):
    x = 0
    for f in os.listdir(dir):
        x += os.path.getsize('%s/%s'%(dir, f))
    return x

def free_space():
    total, used, free = shutil.disk_usage('.')
    return free
