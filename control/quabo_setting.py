#! /usr/bin/env python3

# Functions for setting quabos, including maroc configuration, mask configuration...
# It should be convenient to use these functions for debugging.
# The default quabo config file is quabo_config.txt.abs
import quabo_driver
import json

HV_OFFSET = 1.073
# HV_FACTOR = 2.5/2**16*(301/(4.99+4.99))
HV_FACTOR = 0.0011453

class QUABO_SETTING(object):
    '''
    Class: This class contains low level functions for setting quabos
    '''
    def __init__(self, ip = '192.168.3.248', config_file = 'quabo_config.txt'):
        self.ip = ip
        self.config = quabo_driver.parse_quabo_config_file(config_file)
    
    def set_dac1(self, dac):
        self.config['DAC1'] = '%d,%d,%d,%d'%(dac[0], dac[1], dac[2], dac[3])
    
    def set_dac2(self, dac):
        self.config['DAC2'] = '%d,%d,%d,%d'%(dac[0], dac[1], dac[2], dac[3])
    
    def set_gain(self, i, gain):
        tag = 'GAIN%d'%i
        self.config[tag] = '%d,%d,%d,%d'%(
            gain[0], gain[1],
            gain[2], gain[3]
        )

    def set_fpga_mask(self, i, mask):
        tag = 'CHANMASK_%d'%i
        self.config[tag] = '0x%x'%(mask)

    def set_goe_mask(self, mask):
        self.config['GOEMASK'] = '%d'%(mask)
    
    def set_mask_or1(self, i, mask):
        tag = 'MASKOR1_%d'%(i)
        self.config[tag] = '%d, %d, %d, %d'%(
            mask[0], mask[1],
            mask[2], mask[3]
        )
    
    def set_mask_or2(self, i, mask):
        tag = 'MASKOR2_%d'%(i)
        self.config[tag] = '%d, %d, %d, %d'%(
            mask[0], mask[1],
            mask[2], mask[3]
        )
    
    def set_d1_d2(self, d1_d2):
        tag = 'D1_D2'
        self.config[tag] = '%d, %d, %d, %d'%(
            d1_d2[0], d1_d2[1], 
            d1_d2[2], d1_d2[3]
        )
    
    def set_acq_mode(self, acq):
        self.config['ACQMODE'] = '0x%x'%(acq)

    def set_acq_int(self, integration):
        self.config['ACQINT'] = '%d'%(integration - 1)
    
    def set_hv(self, i, vol):
        tag = 'HV_%d'%(i)
        hv_val = int((vol + HV_OFFSET) / HV_FACTOR)
        self.config[tag] = '%d'%(hv_val)

    # config maroc
    def send_maroc_config(self):
        quabo = quabo_driver.QUABO(self.ip)
        config = self.config.copy()
        quabo.send_maroc_params(config)
        quabo.close()

    # config mask
    def send_mask_config(self):
        quabo = quabo_driver.QUABO(self.ip)
        config = self.config.copy()
        for i in range(9):
            config['CHANMASK_'+str(i)] = int(config['CHANMASK_'+str(i)], 16)
        config['GOEMASK'] = int(config['GOEMASK'], 16)
        quabo.send_trigger_mask(config)
        quabo.send_goe_mask(config)
        quabo.close()
    
    def send_hv(self):
        hv_vals = [0, 0, 0, 0]
        for i in range(4):
            tag = 'HV_%d'%(i)
            hv_vals[i] = int(self.config[tag])
        quabo = quabo_driver.QUABO(self.ip)
        quabo.hv_set(hv_vals)
        quabo.close()

    def send_acq_config(self, params):
        quabo = quabo_driver.QUABO(self.ip)
        quabo.send_daq_params(params)
        quabo.close()

    def write_config(self, fn = 'quabo_config.json'):
        with open(fn, 'w') as f:
            json.dump(self.config, f, indent=4)
    
