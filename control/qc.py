#! /usr/bin/env python3

# cmdline interface to Quabo commands
#
# qc.py ip_addr cmd1 arg ... arg cmd2 arg ... arg
#       do a sequence of commands
#
# qc.py
#       prompt for IP addr and commands from stdin
#
# See https://github.com/panoseti/panoseti/wiki/Quabo-command-line-interface

import quabo_driver, util
import sys

config_filename = "./config/quabo_config.txt"
baseline_filename = "./quabo_baseline.csv"

# do a command; get words from start of list (which may have other commands)
# return true if quit command
#
def do_cmd(quabo, words):
    cmd = words.pop(0)
    if cmd == 'q':
        return True
    elif cmd == 'M':
        quabo.send_maroc_params_file()
    elif cmd == 'B':
        quabo.calibrate_ph_baseline(baseline_filename)
    elif cmd == 'V':
        quabo.hv_config(config_filename)
    elif cmd == 'v':
        chan = int(words.pop(0))
        value = int(words.pop(0))
        quabo.hv_chan(chan, value)
    elif cmd == 'VV':
        quabo.hv_zero()
    elif cmd == 'A':
        quabo.send_acq_parameters_file()
    elif cmd == 'AM':
        image = int(words.pop(0))
        image_us = int(words.pop(0))
        ph = int(words.pop(0))
        bl_subtract = int(words.pop(0))
        am = quabo_driver.DAQ_PARAMS(image, ph, bl_subtract, image_us)
        quabo.send_daq_params(am)
    elif cmd == 'T':
        quabo.send_trigger_mask()
    elif cmd == 'R':
        quabo.reset()
    elif cmd == 'ST':
        steps = int(words.pop(0))
        quabo.focus(steps)
    elif cmd == 'SHO':
        quabo.shutter(False)
    elif cmd == 'SHC':
        quabo.shutter(True)
    elif cmd == 'FAN':
        speed = int(words.pop(0))
        quabo.fan(speed)
    elif cmd == 'SHO_NEW':
        quabo.shutter_new(False)
    elif cmd == 'SHC_NEW':
        quabo.shutter_new(True)
    elif cmd == 'LF0':
        quabo.lf(0)
    elif cmd == 'LF1':
        quabo.lf(1)
    elif cmd == 'HK':
        data = quabo.read_hk_packet()
        if data:
            util.print_binary(data)
        else:
            print('no HK packet')
    elif cmd == 'IM-PH-IP':
        quabo.data_packet_destination(words.pop(0))
    elif cmd == 'HK-IP':
        quabo.hk_packet_destination(words.pop(0))
    else:
        print('unknown command %s'%cmd)

    return False

# command line version: do one or more commands from argv
#
def cmdline():
    sys.argv.pop(0)
    ip_addr = sys.argv.pop(0)
    quabo = quabo_driver.QUABO(ip_addr, config_filename)
    while len(sys.argv) > 0:
        do_cmd(quabo, sys.argv)
    quabo.close()

def print_cmds():
    print('''Enter
    "M" to load only the MAROC setup parameters from quabo_config.txt,
    "B" to calibrate PH Baseline,
    "V" to load only the HV values from quabo_config.txt,
    "v chan value" to adjust individual HV values (0..3, 0..65535),
    "VV" to turn off all HVs,
    "A" to load only the acquisition mode parameters from quabo_config.txt,
    "AM image(0/1) image_us ph(0/1) bl_subtract(0/1): set acq mode params,
    "T" to load the trigger mask values from the quabo_config.txt file,
    "R" to send a system reset,
    "ST steps" to move the focus stepper (1..50000; 0 to recalibrate),
    "SHO" to open shutter with previous firmware(<= V11.1),
    "SHC" to close shutter with previous firmware(<= V11.1),
    "FAN speed" to set fan speed (0..15),
    "SHO_NEW" to open shutter with new firmware(> V11.1),
    "SHC_NEW" to close shutter with new firmware(> V11.1),
    "LF0" to select Led Flasher0 on mobo with new firmware(>= V11.8)
    "LF1" to select Led Flasher1 on mobo with new firmware(>= V11.8)
    "HK": read housekeeping packet
    "IM-PH-IP ip_addr": set destination addr for data packets
    "HK-IP ip_addr": set destination addr for HK packets
    or "q" to quit''')

# interpreter version: prompt for commands
#
def interpreter():
    print("IP address of quabo: ")
    ip_addr = input()
    quabo = quabo_driver.QUABO(ip_addr, config_filename)
    while True:
        print_cmds()
        line = input()
        words = line.split(' ')
        if do_cmd(quabo, words):
            break
    quabo.close()

if len(sys.argv) > 1:
    cmdline()
else:
    interpreter()
