

import sys
import json
import time
import requests

"""
TODO: Add code to add updates to the redis -> influxdb pipeline + add grafana displays for each device.
TODO: Add additional functions to implement other behavior, like starting 
"""
class ControlNode:
    def __init__(self, config):
        self.name = config["name"]
        self.ip_addr = config["ip_addr"]
        self.update_interval_seconds = int(config["update_interval_seconds"]) # TODO: add error checking etc.
        self.device_nodes = []  # This will store DeviceNodeClient objects

    def add_device_node(self, device_node_config):
        """Adds a client device node to this control node."""
        self.device_nodes.append(device_node_config)

    def update_device_status(self):
        """ Retrieve the status of each device node.
        :return: Status response from the device node.
        """
        for device in self.device_nodes:
            status = device.get_status()
            if status:
                print(f"Received status from {device.name}: {status}")
                self.process_status(device, status)
            else:
                print(f"No status received from {device.name}")

    def process_status(self, device_node, status):
        """If the dome and mount are misaligned, send a correction command."""
        if device_node.allow_commands:
            # TODO: Implement logic to check misalignment based on status
            misaligned = False  # Example condition TODO: replace with real logic)
            if misaligned:
                command = {"action": "correct_alignment"}  # Example command TODO: replace with real command
                success = device_node.send_command(command)
                if success:
                    print(f"Command sent successfully to {device_node.name}")
                else:
                    print(f"Failed to send command to {device_node.name}")

    def sleep(self):
        time.sleep(self.update_interval_seconds)


    def stop_system(self):
        # TODO: send commands to device nodes to stop their control servers.
        ...

