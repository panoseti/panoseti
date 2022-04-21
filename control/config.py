#! /usr/bin/env python3

# Initialize for (one or more) observing runs
# See usage() for options.
# see matlab/initq.m, startq*.py

firmware_silver_qfp = 'quabo_0200_264489B3.bin'
firmware_silver_bga = 'quabo_0201_2644962F.bin'
firmware_gold = 'quabo_GOLD_23BD5DA4.bin'

import sys, os, subprocess, time
import util, config_file, file_xfer, quabo_driver
from panoseti_tftp import tftpw

sys.path.insert(0, '../util')
import pixel_coords

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
''')
    sys.exit()

# print summary of obs and daq config files
#
def show_config(obs_config, quabo_uids):
    for dome in obs_config['domes']:
        print('dome %s'%dome['num'])
        for module in dome['modules']:
            module_id = module['id']
            ip_addr = module['ip_addr']
            print('   module ID %d'%module_id)
            print('      Mobo serial#: %s'%module['mobo_serialno'])
            for i in range(4):
                quabo_ip = util.quabo_ip_addr(ip_addr, i)
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
            ip_addr = util.quabo_ip_addr(module['ip_addr'], i)
            print('rebooting quabo at %s'%ip_addr)
            x = tftpw(ip_addr)
            x.reboot()

        # wait for pings
        #
        for module in modules:
            if not util.is_quabo_alive(module, quabo_uids, i):
                continue
            ip_addr = util.quabo_ip_addr(module['ip_addr'], i)
            print('waiting for ping of %s'%ip_addr)
            while True:
                if util.ping(ip_addr):
                    break
                time.sleep(1)
            print('pinged %s; reboot done'%ip_addr)

    print('All quabos rebooted')

def do_loads(modules, quabo_uids):
    for module in modules:
        for i in range(4):
            if not util.is_quabo_alive(module, quabo_uids, i):
                continue
            ip_addr = util.quabo_ip_addr(module['ip_addr'], i)
            if util.is_quabo_old_version(module, i):
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
            ip_addr = util.quabo_ip_addr(module['ip_addr'], i)
            if util.ping(ip_addr):
                print("pinged %s"%ip_addr)
            else:
                print("can't ping %s"%ip_addr)

def do_hk_dest(modules):
    my_ip_addr = util.local_ip()
    for module in modules:
        for i in range(4):
            uid = util.quabo_uid(module, quabo_uids, i)
            if uid == '': continue
            ip_addr = util.quabo_ip_addr(module['ip_addr'], i)
            quabo = quabo_driver.QUABO(ip_addr)
            quabo.hk_packet_destination(my_ip_addr)
            quabo.close()

def do_hv_on(modules, quabo_uids, quabo_info, detector_info):
    for module in modules:
        for i in range(4):
            uid = util.quabo_uid(module, quabo_uids, i)
            if uid == '': continue
            qi = quabo_info[uid]
            v = [0]*4
            for j in range(4):
                det_ser = qi['detector_serialno'][j]
                op_voltage = detector_info[det_ser]
                v[j] = int(op_voltage/.014)
            ip_addr = util.quabo_ip_addr(module['ip_addr'], i)
            quabo = quabo_driver.QUABO(ip_addr)
            quabo.hv_set(v)
            quabo.close()
            print('%s: set HV to [%d %d %d %d]'%(
                ip_addr, v[0], v[1], v[2], v[3]
            ))

def do_hv_off(modules, quabo_uids):
    for module in modules:
        for i in range(4):
            uid = util.quabo_uid(module, quabo_uids, i)
            if uid == '': continue
            v = [0]*4
            ip_addr = util.quabo_ip_addr(module['ip_addr'], i)
            quabo = quabo_driver.QUABO(ip_addr)
            quabo.hv_set(v)
            quabo.close()
            print('%s: set HV to zero'%ip_addr)

# set the DAC1/DA2/GAIN* params for MAROC chips
#
def do_maroc_config(modules, quabo_uids, quabo_info, data_config):
    gain = float(data_config['gain'])
    do_img = 'image' in data_config.keys()
    do_ph = 'pulse_height' in data_config.keys()

    if do_img and do_ph:
        pe_thresh1 = float(data_config['image']['pe_threshold'])
        pe_thresh2 = float(data_config['pulse_height']['pe_threshold'])
    elif do_img:
        pe_thresh1 = float(data_config['image']['pe_threshold'])
    elif do_ph:
        pe_thresh1 = float(data_config['pulse_height']['pe_threshold'])
    else:
        raise Exception('data_config.json specifies no data products')

    qc_dict = quabo_driver.parse_quabo_config_file('quabo_config.txt')
    for module in modules:
        for i in range(4):
            uid = util.quabo_uid(module, quabo_uids, i)
            if uid == '': continue
            is_qfp = util.is_quabo_old_version(module, i)
            qi = quabo_info[uid]
            serialno = qi['serialno'][3:]
            quabo_calib = config_file.get_quabo_calib(serialno)
            ip_addr = util.quabo_ip_addr(module['ip_addr'], i)

            # compute DAC1[] and possibly DAC2 based on calibration data
            dac1 = [0]*4
            dac2 = [0]*4
            for j in range(4):      # 4 detectors in a quabo
                quad = quabo_calib['quadrants'][j]
                a = quad['a']
                b = quad['b']
                dac1[j] = int(a*gain*pe_thresh1 + b)
                if do_img and do_ph:
                    dac2[j] = int(a*gain*pe_thresh2 + b)
            qc_dict['DAC1'] = '%d,%d,%d,%d'%(dac1[0], dac1[1], dac1[2], dac1[3])
            print('%s: DAC1 = %s'%(ip_addr, qc_dict['DAC1']))
            if do_img and do_ph:
                qc_dict['DAC2'] = '%d,%d,%d,%d'%(
                    dac2[0], dac2[1], dac2[2], dac2[3]
                )
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
                print('%s: %s = %s'%(ip_addr, tag, qc_dict[tag]))

            # send MAROC params to the quabo
            quabo = quabo_driver.QUABO(ip_addr)
            quabo.send_maroc_params(qc_dict)
            quabo.close()

if __name__ == "__main__":
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
    if op == 'reboot':
        do_reboot(modules, quabo_uids)
    elif op == 'loads':
        do_loads(modules, quabo_uids)
    elif op == 'ping':
        do_ping(modules)
    elif op == 'init_daq_nodes':
        file_xfer.copy_hashpipe(daq_config)
    elif op == 'hk_dest':
        do_hk_dest(modules)
    elif op == 'redis_daemons':
        util.start_redis_daemons()
    elif op == 'stop_redis_daemons':
        util.stop_redis_daemons()
    elif op == 'show':
        show_config(obs_config, quabo_uids)
        util.show_redis_daemons()
    elif op == 'hv_on':
        detector_info = config_file.get_detector_info()
        do_hv_on(modules, quabo_uids, quabo_info, detector_info)
    elif op == 'hv_off':
        do_hv_off(modules, quabo_uids)
    elif op == 'maroc_config':
        data_config = config_file.get_data_config()
        do_maroc_config(modules, quabo_uids, quabo_info, data_config)
