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

sys.path.insert(0, '../util')
import config_file

def open_domes(obs_config):
    print('Open the shutters of these domes:')
    for dome in obs_config['domes']:
        print('   ', dome['name'])

def session_start(obs_config, quabo_info, data_config, daq_config, no_hv):
    open_domes(obs_config)

    power.do_all(obs_config, 'on')

    print('waiting 40 secs for quabos to come up')
    time.sleep(40)      # wait for quabos to be pingable.  30 is not enough

    print('getting quabo UIDs')
    get_uids.get_uids(obs_config)
    quabo_uids = config_file.get_quabo_uids()

    modules = config_file.get_modules(obs_config)
    print('rebooting quabos')
    config.do_reboot(modules, quabo_uids)

    print('starting Redis daemons')
    util.start_redis_daemons()

    print('setting hk dest to this computer')
    config.do_hk_dest(modules, quabo_uids)

    if not no_hv:
        print('turning on HV')
        detector_info = config_file.get_detector_info()
        config.do_hv_on(modules, quabo_uids, quabo_info, detector_info)

    print('configuring Marocs')
    config.do_maroc_config(modules, quabo_uids, quabo_info, data_config)

    print('configuring Masks')
    config.do_mask_config(modules, data_config)
    
    print('calibrating PH')
    config.do_calibrate_ph(modules, quabo_uids)

    print('opening shutters')
    config.do_shutter("open")

    print('Copying software to DAQ nodes')
    file_xfer.copy_daq_files(daq_config)

if __name__ == "__main__":
    def main():
        no_hv = False
        i = 1
        argv = sys.argv
        while i<len(argv):
            if argv[i] == '--no_hv':
                no_hv = True
            else:
                raise Exception('bad arg %s'%argv[i])
            i += 1
        session_start(
            config_file.get_obs_config(),
            config_file.get_quabo_info(),
            config_file.get_data_config(),
            config_file.get_daq_config(),
            no_hv
        )
    main()
