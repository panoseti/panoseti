#! /usr/bin/env python3

from panoseti_checksys import checksys
import qstart, config_file

def main():
    # set the quabo to image mode
    qstart.qstart(True)

    # get uart_port, wrs_ip from config file
    obs_config = config_file.get_obs_config()
    gps_port = obs_config['gps_port']
    wrs_ip = obs_config['wr_ip_addr']
    cts = checksys(gps_port, wrs_ip)
    state  = cts.check_time_sync()
    print('PANOSETI Time Sync State: ', state)
    
    # set the quabo to idle mode
    qstart.qstart(False)
    
if __name__=='__main__':
    main()
 
