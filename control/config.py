#! /usr/bin/env python3

# Initialize for (one or more) observing runs
# See usage() for options.
# see matlab/initq.m, startq*.py

firmware_silver_qfp = 'quabo_0206_2846D1AE.bin'
firmware_silver_bga = 'quabo_0207_28514055.bin'
firmware_gold = 'quabo_GOLD_23BD5DA4.bin'

import sys, os, subprocess, time, datetime, json, statistics
import logging
from utils import util, file_xfer
from driver.quabo_tftp import tftpw
from driver import quabo_driver
from utils import pixel_coords
from utils import config_file

from argparse import ArgumentParser

# print summary of obs and daq config files
#
def show_config(obs_config, quabo_uids):
    logger = logging.getLogger('PANOSETI.Config.show_config')
    logger.info('Show config')
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
    #print("This node's IP addr: %s"%util.local_ip())
    config_file.show_daq_assignments(quabo_uids)

def do_reboot(modules, quabo_uids, network_config):
    # need to reboot quabos in order 0..3
    # to parallelize:
    # start reboot of quabo 0 in all modules
    # wait for ping of quabo 0 in all modules (means reboot is done)
    # ... same for quabo 1 etc.
    #
    logger = logging.getLogger('PANOSETI.Config.do_reboot')
    for i in range(4):
        for module in modules:
            if not util.is_quabo_alive(module, quabo_uids, i):
                continue
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
            print('rebooting quabo at %s'%ip_addr)
            ip_ports = util.get_quabo_ip_port(module['ip_addr'], i, network_config)
            real_ip = ip_ports['ip_addr']
            cmd_port = ip_ports['cmd_port']
            reboot_port = ip_ports['reboot_port']
            logger.info('Quabo IP: %s'%ip_addr)
            logger.info('Real IP: %s'%real_ip)
            logger.info('Reboot port: %d'%reboot_port)
            x = tftpw(real_ip, reboot_port)
            x.reboot()

        # wait for pings
        #
        for module in modules:
            if not util.is_quabo_alive(module, quabo_uids, i):
                continue
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
            print('waiting for ping of %s'%ip_addr)
            while True:
                logger.info('ping quabo %s:%d...'%(ip_addr, cmd_port))
                # wait for the reboot
                time.sleep(40)
                if util.ping(real_ip, cmd_port):
                    break
                time.sleep(1)
            print('pinged %s; reboot done'%ip_addr)

    print('All quabos rebooted')

def do_loads(modules, quabo_uids, quabo_info, network_config):
    logger = logging.getLogger('PANOSETI.Config.do_loads')
    # TODO The hard-coded path may not be good
    firmware = config_file.get_firmware_config()
    firmware_silver_qfp = 'firmware/' + firmware['qfp']
    firmware_silver_bga = 'firmware/' + firmware['bga']
    for module in modules:
        for i in range(4):
            if not util.is_quabo_alive(module, quabo_uids, i):
                continue
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
            ip_ports = util.get_quabo_ip_port(module['ip_addr'], i, network_config)
            real_ip = ip_ports['ip_addr']
            port = ip_ports['reboot_port']
            logger.info('Real IP: %s'%real_ip)
            logger.info('Reboot Port: %d', port)
            if util.is_quabo_old_version(module, i, quabo_uids, quabo_info):
                fw = firmware_silver_qfp
                logger.info('Loading firmware: %s'%firmware_silver_qfp)
            else:
                fw = firmware_silver_bga
                logger.info('Loading firmware: %s'%firmware_silver_bga)
            x = tftpw(real_ip, port)
            print('loading %s into %s'%(fw, ip_addr))
            x.put_bin_file(fw)

def do_loadg(modules):
    print("not supported")
    #x.put_bin_file(firmware_gold, 0x0)

def do_ping(modules, network_config, verbose=False):
    logger = logging.getLogger('PANOSETI.Config.do_ping')
    ping_record = {
        "ping_true": [],
        "ping_false": []
    }
    for module in modules:
        for i in range(4):
            ip_ports = util.get_quabo_ip_port(module['ip_addr'], i, network_config)
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
            real_ip = ip_ports['ip_addr']
            port = ip_ports['cmd_port']
            logger.info('Real IP: %s'%real_ip)
            logger.info('Cmd Port: %d', port)
            if util.ping(real_ip, port):
                ping_record["ping_true"].append(ip_addr)
            else:
                ping_record["ping_false"].append(ip_addr)
    if verbose:
        for ip in ping_record["ping_true"]:
            print("pinged %s" % ip)
        for ip in ping_record["ping_false"]:
            print("can't ping %s" % ip)
    return ping_record

def do_hk_dest(modules, quabo_uids, daq_config, network_config):
    logger = logging.getLogger('PANOSETI.Config.do_hk_dest')
    headnode_ip_addr = daq_config['head_node_ip_addr']
    logger.info('Head node IP: %s'%headnode_ip_addr)
    for module in modules:
        for i in range(4):
            uid = util.quabo_uid(module, quabo_uids, i)
            if uid == '': continue
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
            ip_ports = util.get_quabo_ip_port(module['ip_addr'], i, network_config)
            real_ip = ip_ports['ip_addr']
            cmd_port = ip_ports['cmd_port']
            logger.info('Quabo IP: %s'%ip_addr)
            logger.info('Real IP: %s'%real_ip)
            logger.info('Cmd Port: %d'%cmd_port)
            quabo = quabo_driver.QUABO(real_ip, cmd_port)
            quabo.hk_packet_destination(headnode_ip_addr)
            quabo.close()

def do_hv_on(modules, quabo_uids, quabo_info, detector_info, network_config, verbose=False):
    logger = logging.getLogger('PANOSETI.Config.do_hv_on')
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
            ip_ports = util.get_quabo_ip_port(module['ip_addr'], i, network_config)
            real_ip = ip_ports['ip_addr']
            cmd_port = ip_ports['cmd_port']
            logger.info('Quabo IP: %s'%ip_addr)
            logger.info('Real IP: %s'%real_ip)
            logger.info('Cmd Port: %d'%cmd_port)
            quabo = quabo_driver.QUABO(real_ip, cmd_port)
            quabo.hv_set(v)
            quabo.close()
            if verbose:
                print('%s: set HV to [%d %d %d %d]'%(
                    ip_addr, v[0], v[1], v[2], v[3]
                ))

def do_hv_off(modules, quabo_uids, network_config):
    logger = logging.getLogger('PANOSETI.Config.do_hv_off')
    for module in modules:
        for i in range(4):
            uid = util.quabo_uid(module, quabo_uids, i)
            if uid == '': continue
            v = [0]*4
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
            ip_ports = util.get_quabo_ip_port(module['ip_addr'], i, network_config)
            real_ip = ip_ports['ip_addr']
            cmd_port = ip_ports['cmd_port']
            logger.info('Quabo IP: %s'%ip_addr)
            logger.info('Real IP: %s'%real_ip)
            logger.info('Cmd Port: %d'%cmd_port)
            quabo = quabo_driver.QUABO(real_ip, cmd_port)
            quabo.hv_set(v)
            quabo.close()
            print('%s: set HV to zero'%ip_addr)

# set the DAC1/DA2/GAIN* params for MAROC chips
#
def do_maroc_config(modules, quabo_uids, quabo_info, data_config, obs_config, daq_config, network_config, verbose=False):
    logger = logging.getLogger('PANOSETI.Config.do_maroc_config')
    no_cali = False
    gain = float(data_config['gain'])
    do_img = 'image' in data_config.keys()
    do_ph = 'pulse_height' in data_config.keys()

    if do_img:
        pe_thresh1 = float(data_config['image']['pe_threshold'])
    if do_ph:
        pe_thresh2 = float(data_config['pulse_height']['pe_threshold'])
    if not do_img and not do_ph:
        raise Exception('data_config.json specifies no data products')

    qc_dict = quabo_driver.parse_quabo_config_file('driver/quabo_config.txt')
    for module in modules:
        for i in range(4):
            uid = util.quabo_uid(module, quabo_uids, i)
            if uid == '': continue
            is_qfp = util.is_quabo_old_version(module, i, quabo_uids, quabo_info)
            try:
                qi = quabo_info[uid]
            except:
                qi = quabo_info['default']
                is_qfp = False
                no_cali = True
            serialno = qi['serialno'][3:]
            # try to find the detector overvoltage in data_config.json
            # if we can't find it, we will use 3v by default.
            try:
                detovervol = data_config['detector_overvoltage']
            except:
                detovervol = 3
            # We have different calibration files for different modes: image alone and image/ph together
            # so we have to specifiy the mode here.
            # TODO: If it's PH alone, what calibration file should we use?
            if do_img and not do_ph:
                op_mode = 'img'
            else:
                op_mode = 'ph'
            quabo_calib = config_file.get_quabo_calib(serialno, detovervol, op_mode)
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
            # set D1_D2 based on the two_pixel_trigger and three_pixel_trigger in data_config.json
            do_two_pixel_trigger = False
            do_three_pixel_trigger = False
            if do_ph:
                if 'two_pixel_trigger' in data_config['pulse_height']:
                    do_two_pixel_trigger = data_config['pulse_height']['two_pixel_trigger']
                if 'three_pixel_trigger' in data_config['pulse_height']:
                    do_three_pixel_trigger = data_config['pulse_height']['three_pixel_trigger']
            # if using 2/3 pixel trigger, D1_D2 should be set to 1,1,1,1
            if do_two_pixel_trigger or do_three_pixel_trigger:
                qc_dict['D1_D2'] = '%d,%d,%d,%d'%(1,1,1,1)
            if verbose:
                print('%s: %s = %s'%(ip_addr, 'D1_D2', qc_dict['D1_D2']))
            # send MAROC params to the quabo
            ip_ports = util.get_quabo_ip_port(module['ip_addr'], i, network_config)
            real_ip = ip_ports['ip_addr']
            cmd_port = ip_ports['cmd_port']
            logger.info('Quabo IP: %s'%ip_addr)
            logger.info('Real IP: %s'%real_ip)
            logger.info('Cmd Port: %d'%cmd_port)
            quabo = quabo_driver.QUABO(real_ip, cmd_port)
            # For ph mode, we seem to have a bug in firmware.
            # we need to set DAC2 to low, and make the quabos send out data first.
            if do_ph:
                tmp = [0] * 4
                # set the DAC2 value very low, 5.5 pe
                for j in range(4):      # 4 detectors in a quabo
                    quad = quabo_calib['quadrants'][j]
                    ah = quad['ah']
                    bh = quad['bh']
                    tmp[j] = int(ah*gain*5.5 + bh)
                qc_dict['DAC2'] = '%d,%d,%d,%d'%(tmp[0],tmp[1],tmp[2],tmp[3])
                quabo.send_maroc_params(qc_dict)
                # make the quabos send out some ph packets
                daq_start = quabo_driver.DAQ_PARAMS(
                        do_image=False,
                        image_us=4999,
                        image_8bit=False,
                        do_ph=True,
                        bl_subtract=True
                    )
                daq_stop = quabo_driver.DAQ_PARAMS(False, 0, False, False, False)
                # This IP is not important, so I put a static IP here.
                # It's just for generating a ph packet
                #daq_node_ip_addr = daq_config['head_node_ip_addr']
                #quabo.data_packet_destination(daq_node_ip_addr)
                quabo.send_daq_params(daq_start)
                time.sleep(1)
                quabo.send_daq_params(daq_stop)
                # set the DAC2 values back
                qc_dict['DAC2'] = '%d,%d,%d,%d'%(dac2[0], dac2[1], dac2[2], dac2[3])
            if no_cali:
                print('**************************************************************************')
                print('Warning: No calibration data for the board with UID: %s'%uid)
                print('         Using default calibration data.')
                print('**************************************************************************')
                logger.warning('No calibration data: UID -%s'%uid)
            quabo.send_maroc_params(qc_dict)
            quabo.write_maroc_config(qc_dict, '%s_%s.json'%('tmp/quabo_config',quabo.ip_addr))
            quabo.close()

# set CHANMASK and GOEMASK for modules
#
def do_mask_config(modules, data_config, network_config, verbose=False):
    logger = logging.getLogger('PANOSETI.Config.do_mask_config')
    qc_dict = quabo_driver.parse_quabo_config_file('quabo_config.txt')
    do_ph = 'pulse_height' in data_config.keys()
    qc_dict['GOEMASK'] = int(qc_dict['GOEMASK'], 16)
    for i in range(9):
        qc_dict['CHANMASK_'+str(i)] = int(qc_dict['CHANMASK_'+str(i)], 16)
    if do_ph:
        # config CHANMASK_8 for any_trigger
        # if we use anytrigger mode, bit8 in CHANMASK_8 should be set to 1 
        if 'any_trigger' in data_config['pulse_height']:
            qc_dict['CHANMASK_8'] = qc_dict['CHANMASK_8'] & 0x0ff
        else:
            qc_dict['CHANMASK_8'] = qc_dict['CHANMASK_8'] | (0x100)
        
        # config GOEMASK for 2/3 pixel_trigger
        # if we use 3 pixel trigger, GOEMASK should be 1, CHANMASK_8 should be 0x0ff or 0x1ff
        if 'three_pixel_trigger' in data_config['pulse_height']:
            if data_config['pulse_height']['three_pixel_trigger']:
                qc_dict['CHANMASK_8'] = qc_dict['CHANMASK_8'] | 0xff
                qc_dict['GOEMASK'] = qc_dict['GOEMASK'] & 0x1
        # if we use 2 pixel trigger, GOEMASK should be 2, CHANMASK_8 should be 0x0ff or 0x1ff
        if 'two_pixel_trigger' in data_config['pulse_height']:
            if data_config['pulse_height']['two_pixel_trigger']:
                qc_dict['CHANMASK_8'] = qc_dict['CHANMASK_8'] | 0xff
                qc_dict['GOEMASK'] = qc_dict['GOEMASK'] & 0x2

    for module in modules:
        for i in range(4):
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
            for tag in ['CHANMASK_8', 'GOEMASK']:
                if verbose:
                    print('%s: %s = 0x%x'%(ip_addr, tag, qc_dict[tag]))
            # send MASK params to the quabo
            ip_ports = util.get_quabo_ip_port(module['ip_addr'], i, network_config)
            real_ip = ip_ports['ip_addr']
            cmd_port = ip_ports['cmd_port']
            logger.info('Quabo IP: %s'%ip_addr)
            logger.info('Real IP: %s'%real_ip)
            logger.info('Cmd Port: %d'%cmd_port)
            quabo = quabo_driver.QUABO(real_ip, cmd_port)
            quabo.send_trigger_mask(qc_dict)
            quabo.send_goe_mask(qc_dict)
            quabo.close()

# compute PH baselines on quabos and write to file
#
def do_calibrate_ph(modules, quabo_uids, network_config):
    logger = logging.getLogger('PANOSETI.Config.do_calibrate_ph')
    quabos = []
    for module in modules:
        for i in range(4):
            uid = util.quabo_uid(module, quabo_uids, i)
            if uid == '': continue
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
            ip_ports = util.get_quabo_ip_port(module['ip_addr'], i, network_config)
            real_ip = ip_ports['ip_addr']
            cmd_port = ip_ports['cmd_port']
            logger.info('Quabo IP: %s'%ip_addr)
            logger.info('Real IP: %s'%real_ip)
            logger.info('Cmd Port: %d'%cmd_port)
            quabo = quabo_driver.QUABO(real_ip, cmd_port)
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
    # create a tmp directory
    baseline_file = config_file.quabo_ph_baseline_filename
    os.makedirs(os.path.dirname(baseline_file), exist_ok=True)
    with open(baseline_file, "w") as f:
        f.write(json.dumps(x, indent=4))


# show summary statistics for the PH baseline calibrations of each quabo
def do_show_ph_baselines(quabo_uids):
    logger = logging.getLogger('PANOSETI.Config.do_show_ph_baselines')
    logger.info('Show PH baseline')
    quabo_ph_baselines = config_file.get_quabo_ph_baselines()
    msg = f"Creation date: {quabo_ph_baselines['date']}\n"
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            module_ip_addr = module['ip_addr']
            msg += f'module {module_ip_addr}:\n'
            for quabo_index in range(4):
                quabo_num = config_file.get_boardloc(module_ip_addr, quabo_index)
                quabo_uid = module['quabos'][quabo_index]['uid']
                quabo_baselines = None
                for q in quabo_ph_baselines['quabos']:
                    if q['uid'] == quabo_uid:
                        quabo_baselines = q
                if quabo_baselines is None:
                    msg += f'\tquabo {quabo_num}: found no ph baseline data\n'
                else:
                    coefs = quabo_baselines['coefs']
                    mean = statistics.mean(coefs)
                    median = statistics.median(coefs)
                    stdev = statistics.stdev(coefs)
                    msg += f'\tquabo {quabo_num: 5}: mean={round(mean, 2): 7}, ' \
                           f'median={round(median, 2): 7}, stdev={round(stdev, 2): 7},' \
                           f' min={min(coefs): 5}, max={max(coefs): 5}\n'
    print(msg)



# compute available recording time, given data config and free disk space.
# If verbose, show details
#
def do_disk_space(data_config, daq_config, verbose=False):
    logger = logging.getLogger('PANOSETI.Config.do_disk_space')
    logger.info('Check disk space.')
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
    # TODO: this is hard-coded??
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


def do_shutter(action):
    if action == "open":
        os.system("tools/shutter.py --open")
    elif action == "close":
        os.system("tools/shutter.py --close")



def main():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logfile = 'logs/config.log'
    util.create_logger(logfile, 'PANOSETI.Config', 'a')
    logger = logging.getLogger('PANOSETI.Config')
    logger.info('************************************')
    parser = ArgumentParser(prog=os.path.basename(__file__), allow_abbrev=False)
    parser.add_argument('--show', dest='show', action='store_true', default=False,
                        help='Show list of domes/modules/quabos.')
    parser.add_argument('--ping', dest='ping', action='store_true', default=False,
                        help='Ping quabos.')
    parser.add_argument('--reboot', dest='reboot', action='store_true', default=False,
                        help='Reboot quabos.')
    parser.add_argument('--loads', dest='loads', action='store_true', default=False,
                        help='Load silver firmware in quabos.')
    parser.add_argument('--init_daq_nodes', dest='init_daq_nodes', action='store_true', default=False,
                        help='Copy software to daq nodes.')
    parser.add_argument('--hk_dest', dest='hk_dest', action='store_true', default=False,
                        help='Set the dest IP for HK packet.')
    parser.add_argument('--redis_daemons', dest='redis_daemons', action='store_true', default=False,
                        help='Start daemons to populate Redis with HK/GPS/WR data, and to copy data from Redis to InfluxDB.')
    parser.add_argument('--stop_redis_daemons', dest='stop_redis_daemons', action='store_true', default=False,
                        help='Stop the above.')
    parser.add_argument('--hv_on', dest='hv_on', action='store_true', default=False,
                        help='Enable detectors.')
    parser.add_argument('--hv_off', dest='hv_off', action='store_true', default=False,
                        help='Disable detectors.')
    parser.add_argument('--maroc_config', dest='maroc_config', action='store_true', default=False,
                        help='Configure MAROCs based on data_config.json and quabo_calib_*.json.')
    parser.add_argument('--mask_config', dest='mask_config', action='store_true', default=False,
                        help='Configure masks based on data_config.json.')
    parser.add_argument('--calibrate_ph', dest='calibrate_ph', action='store_true', default=False,
                        help='Run PH baseline calibration on quabos and write to file')
    parser.add_argument('--show_ph_baselines', dest='show_ph_baselines', action='store_true', default=False,
                        help='Show PH baseline calibration summary statistics')
    parser.add_argument('--shutter_open', dest='shutter_open', action='store_true', default=False,
                        help='Open all module shutters')
    parser.add_argument('--shutter_close', dest='shutter_close', action='store_true', default=False,
                        help='Close all module shutters')
    parser.add_argument('--disk_space', dest='disk_space', action='store_true', default=False,
                        help='Check the disk_space.')
    # we need one option at least
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    # load config files
    obs_config = config_file.get_obs_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()
    daq_config = config_file.get_daq_config()
    quabo_info = config_file.get_quabo_info()
    network_config = config_file.get_network_config()
    util.attach_daq_config(daq_config, network_config)
    config_file.associate(daq_config, quabo_uids)
    data_config = config_file.get_data_config()

    # do the tasks
    if args.reboot:
        do_reboot(modules, quabo_uids, network_config)
        do_hk_dest(modules, quabo_uids, daq_config, network_config)
    elif args.loads:
        do_loads(modules, quabo_uids, quabo_info, network_config)
    elif args.ping:
        do_ping(modules, network_config, verbose=True)
    elif args.init_daq_nodes:
        logger = logging.getLogger('PANOSETI.Config.init_daq_nodes')
        logger.info('Init daq nodes.')
        file_xfer.copy_daq_files(daq_config)
    elif args.hk_dest:
        do_hk_dest(modules, quabo_uids, daq_config, network_config)
    elif args.redis_daemons:
        logger = logging.getLogger('PANOSETI.Config.start_redis_daemons')
        logger.info('Start redis daemons.')
        util.start_redis_daemons()
    elif args.stop_redis_daemons:
        logger = logging.getLogger('PANOSETI.Config.stop_redis_daemons')
        logger.info('Stop redis daemons.')
        util.stop_redis_daemons()
    elif args.show:
        show_config(obs_config, quabo_uids)
        util.show_redis_daemons()
    elif args.hv_on:
        detector_info = config_file.get_detector_info()
        do_hv_on(modules, quabo_uids, quabo_info, detector_info, network_config, True)
    elif args.hv_off:
        do_hv_off(modules, quabo_uids, network_config)
    elif args.maroc_config:
        do_maroc_config(modules, quabo_uids, quabo_info, data_config, obs_config, daq_config, network_config, True)
    elif args.mask_config:
        do_mask_config(modules, data_config, network_config, True)
    elif args.calibrate_ph:
        do_calibrate_ph(modules, quabo_uids, network_config)
    elif args.disk_space:
        do_disk_space(data_config, daq_config, True)
    elif args.shutter_open:
        do_shutter("open")
    elif args.shutter_close:
        do_shutter("close")
    elif args.show_ph_baselines:
        do_show_ph_baselines(quabo_uids)

if __name__ == "__main__":
    main()
