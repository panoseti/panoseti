#! /usr/bin/env python3

"""Control script for dome control system."""
import argparse
import sys
from datetime import datetime
import json
from pathlib import Path

from dome_control_lib import load_dome_control_config
from device_node import DeviceNodeClient
from control_node import ControlNode


def show_config(args):
    config = load_dome_control_config()
    print(json.dumps(config, indent=4))


def setup_control_node(config, control_node_name="headnode"):
    # TODO: move this to the ControlNode class
    # Create control nodes
    assert control_node_name in config
    control_node = ControlNode(config["control_nodes"][control_node_name])

    # Create device nodes and assign them to their respective control node
    for dn in config["device_nodes"]:
        control_node_name = dn["control_node_name"]
        if control_node_name == control_node_name:
            device_node_client = DeviceNodeClient(dn)
            # TODO: make a remote script to startup device nodes on respective computers.
            control_node.add_device_node(device_node_client)
    return control_node


def start_system():
    """Control node main loop. Run this on the control node."""
    config = load_dome_control_config()

    # Set up the system (control nodes and device nodes)
    control_node = setup_control_node(config) # TODO: add setup scripts to run on headnode and device nodes.
    # TODO: add code to startup DeviceNodeServer processes on all required computers.

    # Control node main loop.
    # TODO: move this loop to subprocess running a ControlNode instance.
    # TODO: add signaling logic so
    try:
        while True:
            control_node.update_device_status()
            control_node.sleep()
    except KeyboardInterrupt:
        print("Stopping system")
        control_node.stop_system()


if __name__ == '__main__':
    # create the top-level parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    # create parser for list command
    parser_show_config = subparsers.add_parser('print-config', description='print json config.')
    parser_show_config.set_defaults(func=show_config)

    # create parser for the init command
    parser_init = subparsers.add_parser('start',
                                        description='Start the system')
    parser_init.set_defaults(func=start_system)

    args = parser.parse_args()
    args.func(args)
