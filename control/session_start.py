#! /usr/bin/env python3

# start an "observing session":
# - open domes (TBD)
# - power on relevant modules
# - wait for quabos to come up
# - get quabo UIDs
# - reboot quabos
# - turn on HV (using levels from quabo config files)
# - set gain params of Marocs
# - do PH baseline calibration
# - start the Redis daemons
# - copy software to DAQ nodes

import sys, time, os

import config, power, get_uids, util, file_xfer

import skymap_helper

sys.path.insert(0, '../util')
import config_file

def open_domes(obs_config):
    print('Open the shutters of these domes:')
    for dome in obs_config['domes']:
        print('   ', dome['name'])

def session_start(obs_config, quabo_info, data_config, daq_config, no_hv, stage = 'power_on'):
    modules = config_file.get_modules(obs_config)
    
    # power on the telescopes
    if stage == 'poweron':
        stage = 'ping'
        open_domes(obs_config)
        power.do_all(obs_config, 'on')
        print('waiting 60 secs for quabos to come up')
        time.sleep(60)      # wait for quabos to be pingable.  30 is not enough

    # Wait until all quabos are pingable
    if stage == 'ping':
        stage = 'get_uids'
        all_pinged = False
        while not all_pinged:
            print("pinging quabos...")
            ping_record = config.do_ping(modules, True)
            if len(ping_record["ping_false"]) == 0:
                print("pinged all quabos!")
                all_pinged = True
            else:
                print("failed to ping all quabos. retrying in 5 seconds...")
                time.sleep(5)
    if stage == 'get_uids':
        stage = 'reboot'
        print('getting quabo UIDs')
        get_uids.get_uids(obs_config)
        quabo_uids = config_file.get_quabo_uids()

    if stage == 'reboot':
        stage = 'hk_dest'
        modules = config_file.get_modules(obs_config)
        print('rebooting quabos')
        config.do_reboot(modules, quabo_uids)

    if stage == 'hk_dest':
        stage = 'start_redis'
        print('setting hk dest to this computer')
        config.do_hk_dest(modules, quabo_uids)

    if stage == 'start_redis':
        stage = 'hv_on'
        print('starting Redis daemons')
        util.start_redis_daemons()
    
    if stage == 'hv_on':
        stage = 'mask_config'
        if not no_hv:
            print('turning on HV')
            detector_info = config_file.get_detector_info()
            #config.do_hv_on(modules, quabo_uids, quabo_info, detector_info)
            util.start_hv_updater()
            time.sleep(5) # Wait for hv_updater to start

    if stage == 'mask_config':
        stage = 'maroc_config'
        print('configuring Masks')
        config.do_mask_config(modules, data_config, True)

    if stage == 'maroc_config':
        stage = 'calibrate_ph'
        print('configuring Marocs')
        config.do_maroc_config(modules, quabo_uids, quabo_info, data_config, obs_config, daq_config, True)
    
    if stage == 'calibrate_ph':
        stage = 'open_shutters'
        print('calibrating PH')
        config.do_calibrate_ph(modules, quabo_uids)
        config.do_show_ph_baselines(quabo_uids)

    if stage == 'open_shutters':
        stage = 'copy_sw'
        print('opening shutters')
        config.do_shutter("open")

    if stage == 'copy_sw':
        print('Copying software to DAQ nodes')
        file_xfer.copy_daq_files(daq_config)

def print_help():
    print('Usage: ./session_start.py [--no_hv] [--stage <stage>]')
    print('  --no_hv: Do not turn on HV.')
    print('  --stage <stage>: Start the session from the specified stage.')
    print('    Available stages:')
    print('      poweron        : Power on the modules. (default stage)')
    print('      ping           : Wait until all quabos are pingable.')
    print('      get_uids       : Get quabo UIDs.')
    print('      reboot         : Reboot quabos.')
    print('      hk_dest        : Set dest IP for HK packets.')
    print('      start_redis    : Start Redis daemons.')
    print('      hv_on          : Turn on HV.')
    print('      mask_config    : Configure masks.')
    print('      maroc_config   : Configure Marocs.')
    print('      calibrate_ph   : Calibrate PH.')
    print('      open_shutters  : Open the shutters.')
    print('      copy_sw        : Copy software to DAQ nodes.')

if __name__ == "__main__":
    def main():
        # default stage is 'poweron'
        stage = 'poweron'
        no_hv = False
        i = 1
        argv = sys.argv
        while i<len(argv):
            if argv[i] == '--no_hv':
                no_hv = True
            elif argv[i] == '--stage':
                i += 1
                stage = argv[i]
            elif argv[i] == '--help' or argv[i] == '-h':
                print_help()
                return
            else:
                raise Exception('bad arg %s'%argv[i])
            i += 1
        
        # check if obs_comments.txt exists
        if not os.path.exists('obs_comments.txt'):
            print('obs_comments.txt not found\n')
            print('The format of obs_comments.txt is as follows:')
            print('  line 1         : The name of the observer. (eg. "Wei")')
            print('  line 2,3,4...  : comments. (eg. "This is a test observation.")')
            return
        
        session_start(
            config_file.get_obs_config(),
            config_file.get_quabo_info(),
            config_file.get_data_config(),
            config_file.get_daq_config(),
            no_hv,
            stage
        )
        skymap_helper.start_skymap_info_gen()
    main()
