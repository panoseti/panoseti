#! /usr/bin/env python3

# DEPRECATED
# out of date with ublox_control gRPC implementation.
# TODO:

# ublox_control.py [rpi] [cmd]
#
# rpi (or other name): use the "rpi" from obs_config.json
#
# the u-blox chips are connected to RPIs in each dome.
# Use this script to:
#   1. Configure these chips.
#   2. Start / stop Redis metadata logging.
#
# The IP addr of the RPI and the socket # come from a config file
# This can be used as a module or a script.
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import config_file

import argparse

# TODO: add this config to obs_config instead of hardcoding it here.
rpis = {
    "test": {
        "user": "todo",
        "ip_addr": 0,
        "has_f9t": True,
    }
}


def init_ublox_chip(args):
    rpi = rpis[args.rpi]
    print("init_ublox_chip")
    print(f"{args.rpi}: {rpi=}")

def start_ublox_redis(args):
    rpi = rpis[args.rpi]
    print("start_ublox_redis")
    print(f"{args.rpi}: {rpi=}")

def stop_ublox_redis(args):
    rpi = rpis[args.rpi]
    print("stop_ublox_redis")
    print(f"{args.rpi}: {rpi=}")

def test_ublox_redis(args):
    rpi = rpis[args.rpi]
    interval_seconds = args.interval_seconds
    print("test_ublox_redis")
    print(f"{args.rpi}: {rpi=}, {interval_seconds = }")
    

def cli_handler():
    #ubx = obs_config[name]
    

    # create the top-level parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    # init command parser
    parser_init = subparsers.add_parser('init',
                                        description='Configures u-blox device to start sending metadata packets and verifies they are all being received.')
    parser_init.add_argument('rpi', help='specify the rpi', type=str, choices=rpis.keys())
    parser_init.set_defaults(func=init_ublox_chip)

    # start_redis command parser
    parser_start_redis = subparsers.add_parser('start_redis', description='Start Redis metadata updates from a specified RPI.')
    parser_start_redis.add_argument('rpi', help='specify the rpi', type=str, choices=rpis.keys())
    parser_start_redis.set_defaults(func=start_ublox_redis)

    # stop_redis command parser
    parser_stop_redis = subparsers.add_parser('stop_redis', description='Stop Redis metadata updates from a specified RPI.')
    parser_stop_redis.add_argument('rpi', help='specify the rpi', type=str, choices=rpis.keys())
    parser_stop_redis.set_defaults(func=stop_ublox_redis)

    # test command parser
    parser_test_redis = subparsers.add_parser('test_redis', description='Test Redis dataflow from the specified raspberry pi (RPI)')
    parser_test_redis.add_argument('rpi', help='specify the rpi', type=str, choices=rpis.keys())
    parser_test_redis.add_argument("-n", "--interval_seconds", help="number of seconds between test write operations. Default: 1 second", type=int, default=1)
    parser_test_redis.set_defaults(func=test_ublox_redis)

    # Dispatch command action
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    cli_handler()
