
import requests

from flask import Flask, request, jsonify
import threading
import time


class DeviceNodeClient:
    def __init__(self, config):
        self.name = config["name"]
        self.group_id = config["group_id"]
        self.ip_addr = config["ip_addr"]
        self.network_port = config["network_port"]
        self.device_port = config["device_port"]
        self.device_type = config["device_type"]
        self.device_driver = config["device_driver"]
        self.allow_commands = config["allow_commands"]
        self.control_node_name = config["control_node_name"]

        self.base_url = f"http://{self.ip_addr}:{self.network_port}"

    def send_command(self, command):
        """ Send a command to the device node server. """
        url = f"{self.base_url}/command"
        try:
            response = requests.post(url, json=command)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Failed to send command to {self.ip_addr}:{self.network_port} - {e}")
            return None

    def get_status(self):
        """ Retrieve the status of the device node server. """
        url = f"{self.base_url}/status"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Failed to send command to {self.ip_addr}:{self.network_port} - {e}")
            return None


# TODO: Fill in this skeleton code
# TODO: create a remote script to create a subprocess in which to run this server.
"""
TODO: add support for additional functions to do things like:
1) open and close the slie
2) command left and right relative and absolute dome and mount rotations
3) query status of other stuff
"""
# TODO: add a shutdown function to clean up any resources / open device files etc. then exit the process.
class DeviceNodeServer:
    def __init__(self, config):
        self.name = config["name"]
        self.group_id = config["group_id"]
        self.ip_addr = config["ip_addr"]
        self.network_port = config["network_port"]
        self.device_port = config["device_port"]
        self.device_type = config["device_type"]
        self.device_driver = config["device_driver"]
        self.allow_commands = config["allow_commands"]
        self.control_node_name = config["control_node_name"]

        # TODO: add other fields to store data about the device.
        # TODO: make these fields depend on the type of device this server services.
        self.current_position = 0
        self.is_moving = False

        # Create a Flask app
        self.app = Flask(__name__)

        # Define API endpoints
        self.app.add_url_rule('/command', 'command', self.handle_command, methods=['POST'])
        self.app.add_url_rule('/status', 'status', self.get_status, methods=['GET'])

    def handle_command(self):
        """
        Handle incoming commands to move the device.
        """
        data = request.json
        # TODO: Add more actions + types of movements (e.g. rotations, opening dome slit, etc.)
        action = data.get('action')
        target_position = data.get('target_position')

        if action == "move" and target_position is not None:
            # TODO: verify that this works
            threading.Thread(target=self.move_device, args=(target_position,)).start()
            # TODO: add dispatch function to make this behavior more sophisticated.
            # TODO: move the responsibility of sending response to
            return jsonify({"status": "moving", "target_position": target_position}), 200
        else:
            return jsonify({"error": "Invalid command"}), 400

    def move_device(self, target_position):
        """ Simulate moving the device to the target position. """
        self.is_moving = True
        print(f"Moving {self.device_name} to position {target_position}...")
        # TODO: replace move simulation with actual calls to device driver.
        time.sleep(5)
        self.current_position = target_position
        self.is_moving = False
        print(f"{self.device_name} reached position {self.current_position}.")
        # TODO: inform the server about the result of the move action via a response.

    def get_status(self):
        """ Return the current status of the device. """
        response = {
            "device_name": self.device_name,
            "current_position": self.current_position,
            "is_moving": self.is_moving
        }
        return jsonify(response)

    def run(self):
        """ Run the Flask application. """
        self.app.run(host='0.0.0.0', port=self.port)


# Example of running the server (to be executed separately)
if __name__ == "__main__":
    # TODO: turn this into a remote script
    device_server = DeviceNodeServer(device_name="Crocker Mount", port=8080)
    device_server.run()
