#! /usr/bin/env python3

"""Control script for dome control system."""
import argparse
import sys
import socket
from datetime import datetime
import pprint
import json
from pathlib import Path

sys.path.insert(0, '../util')

dome_control_config = 'dome_control_config.json'

def get_dome_control_config():
    with open(dome_control_config, 'r') as fp:
        config = json.load(fp)
    return config

def init(args):
    config = get_dome_control_config()



def show_config(args):
    # pp = pprint.PrettyPrinter(indent=4)
    config = get_dome_control_config()
    print(json.dumps(config, indent=4))
    # pp.pprint(config)


if __name__ == '__main__':
    # create the top-level parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    # create parser for list command
    parser_show_config = subparsers.add_parser('show-config', description='print json config.')
    parser_show_config.set_defaults(func=show_config)

    # create parser for the init command
    parser_init = subparsers.add_parser('init',
                                        description='Configure device node and establish connection to server node.')
    parser_init.add_argument('device',
                             help='specify the device path. example: /dev/ttyS3',
                             type=str)
    parser_init.set_defaults(func=init)

    args = parser.parse_args()
    args.func(args)