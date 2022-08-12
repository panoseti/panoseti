#! /usr/bin/env python3

# Tell a single quabo to send image packets (for timing test)
# You can use this as a script or a module.

import config_file, quabo_driver

def qtest():
    obs_config = config_file.get_obs_config()
    d = obs_config['domes'][0]
    m = d['modules'][0]
    ip_addr = m['ip_addr']
    quabo = quabo_driver.QUABO(ip_addr)
    quabo.send_daq_params(
        quabo_driver.DAQ_PARAMS(
            True, 10000-1, False, False, False
        )
    )
    quabo.close()

if __name__ == "__main__":
    qtest()