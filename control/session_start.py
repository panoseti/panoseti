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

import config, power, get_uids
from utils import config_file, util

from argparse import ArgumentParser

def session_start(obs_config, quabo_info, data_config, daq_config, network_config, no_hv, stage ):

    modules = config_file.get_modules(obs_config)
    # power on the telescopes
    if stage == 'poweron':
        stage = 'get_uids'
        power.do_all(obs_config, 'on')
        print('waiting 60 secs for quabos to boot up')
        time.sleep(60)

    if stage == 'get_uids':
        stage = 'reboot'
        print('getting quabo UIDs')
        get_uids.get_uids(obs_config, network_config)

    if stage == 'reboot':
        stage = 'hk_dest'
        modules = config_file.get_modules(obs_config)
        print('rebooting quabos')
        quabo_uids = config_file.get_quabo_uids()
        status = config.do_reboot(modules, quabo_uids, network_config)
        if status == False:
            print('Reboot Failed.')
            return
        else:
            print('Reboot Successfully.')

    if stage == 'hk_dest':
        stage = 'start_redis'
        print('setting hk dest to this computer')
        quabo_uids = config_file.get_quabo_uids()
        config.do_hk_dest(modules, quabo_uids, daq_config, network_config)

    if stage == 'start_redis':
        stage = 'maroc_config'
        print('starting Redis daemons')
        util.start_redis_daemons()
    
    if stage == 'maroc_config':
        stage = 'mask_config'
        print('configuring Marocs')
        quabo_uids = config_file.get_quabo_uids()
        config.do_maroc_config(modules, quabo_uids, quabo_info, data_config, obs_config, daq_config, network_config, True)

    if stage == 'mask_config':
        stage = 'calibrate_ph'
        print('configuring Masks')
        quabo_uids = config_file.get_quabo_uids()
        config.do_mask_config(modules, data_config, network_config, quabo_uids, True)
    
    if stage == 'calibrate_ph':
        stage = 'open_shutters'
        print('calibrating PH')
        config.do_calibrate_ph(modules, quabo_uids, network_config)
        config.do_show_ph_baselines(quabo_uids)

    # TODO: we need more tests for do_shutter
    # if stage == 'open_shutters':
    #     print('opening shutters')
    #     config.do_shutter("open")

def main():
    parser = ArgumentParser(prog=os.path.basename(__file__), allow_abbrev=False)
    parser.add_argument('--no_hv', dest='no_hv', action='store_true', default=False,
                        help='Turn off HV when running `start.py`.')
    parser.add_argument('--stage', dest='stage', type=str, default='poweron', 
                        choices=['poweron', 'get_uids', 'reboot', 'hk_dest', 'start_redis',
                                 'maroc_config', 'mask_config', 'calibrate_ph', 'show_ph_baselines'],
                        help='The session will start from this stage.')
    # parse the args
    args = parser.parse_args()
    # session start
    session_start(
            config_file.get_obs_config(),
            config_file.get_quabo_info(),
            config_file.get_data_config(),
            config_file.get_daq_config(),
            config_file.get_network_config(),
            args.no_hv,
            args.stage
        )

if __name__ == "__main__":
    main()
