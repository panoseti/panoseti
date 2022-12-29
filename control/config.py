#! /usr/bin/env python3

# Initialize for (one or more) observing runs
# See usage() for options.
# see matlab/initq.m, startq*.py

firmware_silver_qfp = 'quabo_0200_264489B3.bin'
firmware_silver_bga = 'quabo_0201_2644962F.bin'
firmware_gold = 'quabo_GOLD_23BD5DA4.bin'

import sys, os, subprocess, time, datetime, json
import util, file_xfer, quabo_driver
from panoseti_tftp import tftpw

sys.path.insert(0, '../util')
import pixel_coords
import config_file

def usage():
    print('''usage:
--show                  show list of domes/modules/quabos
--ping                  ping quabos
--reboot                reboot quabos
--loads                 load silver firmware in quabos
--init_daq_nodes        copy software to daq nodes
--hk_dest               tell quabos to send HK packets to this node
--redis_daemons         start daemons to populate Redis with HK/GPS/WR data,
                        and to copy data from Redis to InfluxDB
--stop_redis_daemons    stop the above
--hv_on                 enable detectors
--hv_off                disable detectors
--maroc_config          configure MAROCs based on data_config.json
                        and quabo_calib_*.json
--calibrate_ph          run PH baseline calibration on quabos and write to file
''')
    sys.exit()

# print summary of obs and daq config files
#
def show_config(obs_config, quabo_uids):
    for dome in obs_config['domes']:
        print('dome %s'%dome['name'])
        for module in dome['modules']:
            module_id = module['id']
            ip_addr = module['ip_addr']
            print('   module ID %d'%module_id)
            print('      Mobo serial#: %s'%module['mobo_serialno'])
            for i in range(4):
                quabo_ip = config_file.quabo_ip_addr(ip_addr, i)
                print('      quabo %d'%i)
                print('         IP addr: %s'%quabo_ip)
    print("This node's IP addr: %s"%util.local_ip())
    config_file.show_daq_assignments(quabo_uids)

def do_reboot(modules, quabo_uids):
    # need to reboot quabos in order 0..3
    # to parallelize:
    # start reboot of quabo 0 in all modules
    # wait for ping of quabo 0 in all modules (means reboot is done)
    # ... same for quabo 1 etc.
    #
    for i in range(4):
        for module in modules:
            if not util.is_quabo_alive(module, quabo_uids, i):
                continue
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
            print('rebooting quabo at %s'%ip_addr)
            x = tftpw(ip_addr)
            x.reboot()

        # wait for pings
        #
        for module in modules:
            if not util.is_quabo_alive(module, quabo_uids, i):
                continue
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
            print('waiting for ping of %s'%ip_addr)
            while True:
                if util.ping(ip_addr):
                    break
                time.sleep(1)
            print('pinged %s; reboot done'%ip_addr)

    print('All quabos rebooted')

def do_loads(modules, quabo_uids, quabo_info):
    for module in modules:
        for i in range(4):
            if not util.is_quabo_alive(module, quabo_uids, i):
                continue
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
            if util.is_quabo_old_version(module, i, quabo_uids, quabo_info):
                fw = firmware_silver_qfp
            else:
                fw = firmware_silver_bga
            x = tftpw(ip_addr)
            print('loading %s into %s'%(fw, ip_addr))
            x.put_bin_file(fw)

def do_loadg(modules):
    print("not supported")
    #x.put_bin_file(firmware_gold, 0x0)

def do_ping(modules):
    for module in modules:
        for i in range(4):
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
            if util.ping(ip_addr):
                print("pinged %s"%ip_addr)
            else:
                print("can't ping %s"%ip_addr)

def do_hk_dest(modules, quabo_uids):
    my_ip_addr = util.local_ip()
    for module in modules:
        for i in range(4):
            uid = util.quabo_uid(module, quabo_uids, i)
            if uid == '': continue
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
            quabo = quabo_driver.QUABO(ip_addr)
            quabo.hk_packet_destination(my_ip_addr)
            quabo.close()

def do_hv_on(modules, quabo_uids, quabo_info, detector_info, verbose=False):
    for module in modules:
        for i in range(4):
            uid = util.quabo_uid(module, quabo_uids, i)
            if uid == '': continue
            qi = quabo_info[uid]
            v = [0]*4
            for j in range(4):
                det_ser = qi['detector_serialno'][j]
                op_voltage = detector_info[str(det_ser)]
                v[j] = int(op_voltage/.00114)
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
            quabo = quabo_driver.QUABO(ip_addr)
            quabo.hv_set(v)
            quabo.close()
            if verbose:
                print('%s: set HV to [%d %d %d %d]'%(
                    ip_addr, v[0], v[1], v[2], v[3]
                ))

def do_hv_off(modules, quabo_uids):
    for module in modules:
        for i in range(4):
            uid = util.quabo_uid(module, quabo_uids, i)
            if uid == '': continue
            v = [0]*4
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
            quabo = quabo_driver.QUABO(ip_addr)
            quabo.hv_set(v)
            quabo.close()
            print('%s: set HV to zero'%ip_addr)

# set the DAC1/DA2/GAIN* params for MAROC chips
#
def do_maroc_config(modules, quabo_uids, quabo_info, data_config, verbose=False):
    gain = float(data_config['gain'])
    do_img = 'image' in data_config.keys()
    do_ph = 'pulse_height' in data_config.keys()

    if do_img:
        pe_thresh1 = float(data_config['image']['pe_threshold'])
    if do_ph:
        pe_thresh2 = float(data_config['pulse_height']['pe_threshold'])
    if not do_img and not do_ph:
        raise Exception('data_config.json specifies no data products')

    qc_dict = quabo_driver.parse_quabo_config_file('quabo_config.txt')
    for module in modules:
        for i in range(4):
            uid = util.quabo_uid(module, quabo_uids, i)
            if uid == '': continue
            is_qfp = util.is_quabo_old_version(module, i, quabo_uids, quabo_info)
            qi = quabo_info[uid]
            serialno = qi['serialno'][3:]
            quabo_calib = config_file.get_quabo_calib(serialno)
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)

            # compute DAC1[] and possibly DAC2 based on calibration data
            dac1 = [0]*4
            dac2 = [0]*4
            for j in range(4):      # 4 detectors in a quabo
                quad = quabo_calib['quadrants'][j]
                a = quad['a']       # a and b are used for img mode
                b = quad['b']
                ah= quad['ah']      # ah and bh are used for ph mode
                bh= quad['bh']
                if do_img:
                    dac1[j] = int(a*gain*pe_thresh1 + b)
                if do_ph:
                    dac2[j] = int(ah*gain*pe_thresh2 + bh)
            if do_img:
                qc_dict['DAC1'] = '%d,%d,%d,%d'%(dac1[0], dac1[1], dac1[2], dac1[3])
                if verbose:
                    print('%s: DAC1 = %s'%(ip_addr, qc_dict['DAC1'])) 
            if do_ph:
                qc_dict['DAC2'] = '%d,%d,%d,%d'%(dac2[0], dac2[1], dac2[2], dac2[3])
                if verbose:
                    print('%s: DAC2 = %s'%(ip_addr, qc_dict['DAC2']))
            # compute GAIN0[]..GAIN63[] based on calibration data
            # TODO: fix indexing
            maroc_gain = [[0]*4 for i in range(64)]

            for j in range(4):
                for k in range(64):
                    [x, y] = pixel_coords.detector_to_quabo(k, j, is_qfp)
                    delta = quabo_calib['pixel_gain'][x][y]
                    g = int(round(gain*(1+delta)))
                    maroc_gain[k][j] = g
            for k in range(64):
                tag = 'GAIN%d'%k
                qc_dict[tag] = '%d,%d,%d,%d'%(
                    maroc_gain[k][0], maroc_gain[k][1],
                    maroc_gain[k][2], maroc_gain[k][3]
                )
                if verbose:
                    print('%s: %s = %s'%(ip_addr, tag, qc_dict[tag]))

            # send MAROC params to the quabo
            quabo = quabo_driver.QUABO(ip_addr)
            quabo.send_maroc_params(qc_dict)
            time.sleep(0.1)
            quabo.send_trigger_mask()
            quabo.close()

# compute PH baselines on quabos and write to file
#
def do_calibrate_ph(modules, quabo_uids):
    quabos = []
    for module in modules:
        for i in range(4):
            uid = util.quabo_uid(module, quabo_uids, i)
            if uid == '': continue
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
            quabo = quabo_driver.QUABO(ip_addr)
            coefs = quabo.calibrate_ph_baseline()
            quabo.close()
            q = {}
            q['uid'] = uid
            q['coefs'] = coefs
            quabos.append(q)
    x={}
    d = datetime.datetime.utcnow()
    x['date'] = d.isoformat()
    x['quabos'] = quabos;
    with open(config_file.quabo_ph_baseline_filename, "w") as f:
        f.write(json.dumps(x, indent=4))

# compute available recording time, given data config and free disk space.
# If verbose, show details
#
def do_disk_space(data_config, daq_config, verbose=False):
    bps = util.daq_bytes_per_sec_per_module(data_config)
    if verbose:
        print('Data rate per module: %.2f MB/sec'%(bps/1e6))
    nmod_total = 0
    available_hours = 1e9

    # loop over DAQ nodes
    #
    for node in daq_config['daq_nodes']:
        if not node['modules']:
            continue
        nmod = len(node['modules'])
        nmod_total += nmod
        ip_addr = node['ip_addr']
        if verbose:
            print('DAQ node %s: %d modules'%(ip_addr, nmod))

        # get list of volumes on the DAQ node
        #
        j = util.get_daq_node_status(node)
        vols = j['vols']

        # initialize list of module IDs each vol will handle,
        # and find the default volume for this node
        #
        default_vol = None
        for vol in vols.values():
            vol['mods_here'] = []
            if -1 in vol['modules']:
                default_vol = vol

        # loop over module IDs going to this DAQ node,
        # and add them to the mods_here list for the appropriate volume
        #
        for module in node['modules']:
            mid = module['id']
            found = False
            for vol in vols.values():
                if mid in vol['modules']:
                    vol['mods_here'].append(mid)
                    found = True
                    break
            if not found:
                default_vol['mods_here'].append(mid)

        for name in vols.keys():
            vol = vols[name]
            free = vol['free']
            nmods = len(['mods_here'])
            if verbose:
                print('   %s:'%name)
            if nmods:
                t = free/(3600.*bps*nmods)
                if verbose:
                    print('      modules: ', vol['mods_here'])
                    print('      space: %.2fTB (%.2f hours)'%(free/1e12, t))
                if t < available_hours:
                    available_hours = t
            else:
                if verbose:
                    print('      space: %.2fTB'%(free/1e12))

    head_node_vols = json.loads(open("/home/panosetigraph/web/head_node_volumes.json").read())
    hnd = daq_config['head_node_data_dir']
    hnd = os.path.realpath(hnd)
    print('head node:')
    for vol in head_node_vols:
        path = '/home/panosetigraph/web/%s/data'%vol
        path = os.path.realpath(path)
        hfree = util.free_space(path)
        if verbose:
            print('   %s (%s)'%(path, vol))
        t = hfree/(3600*bps*nmod_total)
        if hnd == path:
            if t < available_hours:
                available_hours = t
            if verbose:
                print('      selected for write')
        print('      space: %.2fTB (%.2f hours)'%(hfree/1e12, t))

    if verbose:
        print('---------------\nAvailable recording time: %.2f hours'%available_hours)
    return available_hours

if __name__ == "__main__":
    def main():
        argv = sys.argv
        nops = 0
        i = 1
        while i < len(argv):
            if argv[i] == '--show':
                nops += 1
                op = 'show'
            elif argv[i] == '--reboot':
                nops += 1
                op = 'reboot'
            elif argv[i] == '--loads':
                nops += 1
                op = 'loads'
            elif argv[i] == '--ping':
                nops += 1
                op = 'ping'
            elif argv[i] == '--init_daq_nodes':
                nops += 1
                op = 'init_daq_nodes'
            elif argv[i] == '--hk_dest':
                nops += 1
                op = 'hk_dest'
            elif argv[i] == '--redis_daemons':
                nops += 1
                op = 'redis_daemons'
            elif argv[i] == '--stop_redis_daemons':
                nops += 1
                op = 'stop_redis_daemons'
            elif argv[i] == '--hv_on':
                nops += 1
                op = 'hv_on'
            elif argv[i] == '--hv_off':
                nops += 1
                op = 'hv_off'
            elif argv[i] == '--maroc_config':
                nops += 1
                op = 'maroc_config'
            elif argv[i] == '--calibrate_ph':
                nops += 1
                op = 'calibrate_ph'
            elif argv[i] == '--disk_space':
                nops += 1
                op = 'disk_space'
            else:
                print('bad arg: %s'%argv[i])
                usage()
            i += 1

        if nops == 0:
            usage()
        if nops > 1:
            print('must specify a single op')
            usage()

        obs_config = config_file.get_obs_config()
        modules = config_file.get_modules(obs_config)
        quabo_uids = config_file.get_quabo_uids()
        daq_config = config_file.get_daq_config()
        quabo_info = config_file.get_quabo_info()
        config_file.associate(daq_config, quabo_uids)
        data_config = config_file.get_data_config()
        if op == 'reboot':
            do_reboot(modules, quabo_uids)
            do_hk_dest(modules, quabo_uids)
        elif op == 'loads':
            do_loads(modules, quabo_uids, quabo_info)
        elif op == 'ping':
            do_ping(modules)
        elif op == 'init_daq_nodes':
            file_xfer.copy_daq_files(daq_config)
        elif op == 'hk_dest':
            do_hk_dest(modules, quabo_uids)
        elif op == 'redis_daemons':
            util.start_redis_daemons()
        elif op == 'stop_redis_daemons':
            util.stop_redis_daemons()
        elif op == 'show':
            show_config(obs_config, quabo_uids)
            util.show_redis_daemons()
        elif op == 'hv_on':
            detector_info = config_file.get_detector_info()
            do_hv_on(modules, quabo_uids, quabo_info, detector_info, True)
        elif op == 'hv_off':
            do_hv_off(modules, quabo_uids)
        elif op == 'maroc_config':
            do_maroc_config(modules, quabo_uids, quabo_info, data_config, True)
        elif op == 'calibrate_ph':
            do_calibrate_ph(modules, quabo_uids)
        elif op == 'disk_space':
            do_disk_space(data_config, daq_config, True)
    main()
