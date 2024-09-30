# Design brainstorming

## Random ideas

- Dome controller interface:
    - Rotate by x degrees to the right / left
    - query the current rotational position.l
- Client Head node code:
    1. Accept data updates from the rpi.
    2. Send updates to the dome controller.
- We should abstract away the network details from this interface as they can be expected to change.
    - To make the implementation as flexible as possible, we should not make any assumptions about which computers things are on.
    - So this will be purely network-driven approach to the problem.
- In the future, we will probably have a single RPI controlling domes
- Also, this implementation should be above the layer of the specific mount and dome controllers we use.
    - In the future we may use different mounts and dome technology.
    - To make this work as general as possible we should divide it into a few abstraction layers, similar to an OS structure.
- Thoughts: We should design our system so that we can easily replace sockets with pipes in case we want to run the controller and client program on the same computer.
    - In the future the dome and mount controllers may be accessible to the same computer without needing to be routed through the head node.
    - 
- Device Node (multithreaded?)
    - Config information:
        - Server IP address.
        - Connected devices: dome controller, mount controller, or both.
            - For simplicity, we may want to make a single client process to handle each device.
            - It should be ok for this process to run with single thread:
        - Mode: server-control.
            - In server-control, the clients monitoring the respective dome controllers will respond to server data and movement requests.
    - Client tasks / functions
        - Establish server connection.
        - Query connected device(s) for current status.
        - Send current device status to the server.
        - Listen for commands from the server.
        - Validate and execute movement commands from the server.
    1. Lowest level: device drivers
    2. Highest level: dome movement / data transfer.
        1. We could program the dome scheduler to 
- Control Node (may also be a client node).
    - Config information:
        - Client data (per client)
            - IP address
            - Connected devices (mount, control)
        - 
    - Server tasks:
        - Regularly equest position updates
        - 2) issue commands to the clients.
- Maybe have the same code for control and device nodes?
    - General config for the relation between different nodes:
    - Per unique node in the network:
        - Node IP address
        - Devices it controls (if any).
        - Controller: which node directly controls this node?
        - Reader: which nodes should this node send position data to?

## Configuration file

### General format specification

```json
{
	"name": "Lick",
	"comment": "Default Lick config: just crocker dome",
	"device_nodes": [
		{
			"group_id": "An string common to all device nodes in the same dome. Used to identify which device nodes are related to each other",
			"name": "name of this device node. Ex: dnode_crocker_mount, dnode_crocker_dome",
			"ip_addr": "Alias or fixed IP address of this computer in the network",
			"control_node_name": "name of the control node for this device node",
			"device_type": "As of 9/30/24 must be either MOUNT_CONTROLLER or DOME_CONTROLLER",
			"device_port": "TODO",
			"device_driver": "name of specific device driver for this device.",
			"allow_commands": "whether this node should execute movement commands",
			"comment": "Each device node must be connected to exactly one physical device.",
		},
	],
	"control_nodes": [
		{
			"name": "name for this control node; referenced by device node configs",
			"ip_addr": "Alias or fixed IP address of this computer in the network",
			"update_interval_seconds": "duration in seconds between successive query-position-updates.",
		},
	]
}
```

### Example for Lick:

```json
{
	"name": "Lick",
	"comment": "Default Lick config: just crocker dome",
	"control_nodes": [
		{
			"name": "headnode",
			"ip_addr": "192.168.3.99",
			"update_interval_seconds": 30
		}
	],
	"device_nodes": [
		{
			"group_id": "crocker",
			"name": "rpi_ours",
			"ip_addr": "192.168.3.12",
			"control_node_name": "headnode",
			"device_type": "MOUNT_CONTROLLER",
			"device_port": "TODO",
			"device_driver": "crocker_mount_controller",
			"allow_commands": "false",
			"comment": "Each device node must be connected to exactly one physical device."
		},
		{
			"group_id": "crocker",
			"name": "rpi_jeff",
			"ip_addr": "128.114.176.148",
			"control_node_name": "headnode",
			"device_type": "DOME_CONTROLLER",
			"device_port": "/dev/ttyUSB_DOME",
			"device_driver": "crocker_dome_controller",
			"allow_commands": "true",
			"comment": "Each device node must be connected to exactly one physical device."
		}
	]
}
```

![image.png](Design%20brainstorming%201104d556e62b800a82bbc6e4a5568e3c/image.png)

## Control node class

### Data structures:

- `node_config`: configuration info for this control node, including dict of device nodes (e.g. which mount and dome nodes are connected to each other).
    - Loaded from JSON config file.
- `device_node_connections`: currently established device node connections
- `command_history`: DataFrame of all past commands to all device nodes.
    - Saved in run directory.
- `status_history`: DataFrame of all status information updates from device nodes.
    - Saved in run directory.
- `device_status_queue`: Thread-safe [queue](https://www.notion.so/Design-brainstorming-1104d556e62b800a82bbc6e4a5568e3c?pvs=21) containing unprocessed status updates from every device node.

### Functions:

- `query_device(device_node)`
    - Requests current device status from `device_node`.
- `set_query_schedule(device_node, interval)`
    - Configures `device_node` to send current device status updates every `interval` seconds.
- `get_alignment_command(device_node)`
    - Use the history of status information for `device_node` in `status_history` to compute a command that will align the mount and the dome.
        - Essentially, this code will command the device to move to a new position / orientation.
    - Return this command.
- `send_command(device_node, cmd)`
    - Send a movement command to device_node.
- `process_status_update(status_update struct)`
    - Pop a status update from the queue.
    - Add status update to the redis database for long-term logging in influxdb.
    - Add status update to `status_history`
    - If the status update is a receipt for a previous command `cmd`:
        - Verify the reported new position is within a certain tolerance of the requested position and / or if the mount and dome delta is good.
            - `cmd_status` = “EX_ALIGN_OK” if the command was received and executed AND the new position looks good
            - `cmd_status` = “EX_ALIGN_BAD” if the command was received and executed AND the new position is looks bad.
            - `cmd_status` = “ERROR” if the command receipt indicates an error occurred while trying to do the movement.
        - Update the `cmd_status` field of the entry for `cmd` in `command_history` to
    - Else:
        - Check if the alignment delta between the mount and the dome slit exceeds a certain limit. (Implement hysteresis thresholds here.)
        - If dome movement needs to occur:
            - Get the appropriate movement command `cmd`from `get_alignment_command`.
            - Use `send_command(device_node, cmd)` to request the movement.
            - Add the `cmd` to `command_history` and set the `cmd_status` field to "SENT"

### Server Main Loop (Thread 0)

- 

## Device node class

### Data structures:

- `node_config`:
    - Device type: “dome” or “mount”
    - Device driver interface info:
        - Describes specific device driver this node should use (e.g. Crocker dome controller, Barnard dome controller, etc.)
        - Lists supported interface commands and parameters (both required and optional).
    - Controller node info:
        - ip address
        - permission: allow movement commands?: whether this device node should attempt to execute movement commands provided by this controller node (may not be possible: device driver interface for mount may not have implemented this stuff).
- `control_node_connection`: connection reference (?) to control node.
- `command_queue`:

### Functions:

- `server_connect(server_ip)`:
    - Establish connection to the
- `query_device(device)`: queries the connected device and returns a vector of status information (device-specific).
    - JSON dict: local storage and optionally forward to the control node.
    - Allow control node to request regular updates every X seconds → set a wake timer
        - https://web.archive.org/web/20200319103023/http://effbot.org/zone/thread-synchronization.htm
    - Send status info to the control node (e.g. redis database etc.)
- `cmd_queue_add(command_ID, parameters)`
    - Add new command to the queue
- `run_next_command()`:
    - Do next command in the queue.
    - Returns status of command success or None if the command queue is empty.
- `command_execute(command_ID, parameters)`: send movement commands to connected device.
    - Validate movement command (is it a valid command?)
        - Dome: open shutter etc.
    - Use device driver code to do the movement.
    - Send message about success of the movement (success or failure).
        - Also report the displacement metadata: old position and new position

### Device Main Loop (Thread 0)

- Initialize device node thread (Thread 1)
- Establish connection between device node and control node.
- Listen for server requests:
    - Verify that command is next in the sequence of movement commands provided by control node:
        - Server should provide sequential information with each command so that if network delays cause packets to get switched around, the dome will only execute commands in sequence: if we eventually control mount, some rotation operations may be non-commutative.
    - Add command to client queue.

## Device Driver Interfaces

### Overview

Provides an abstract interface for basic operations related to dome and mount alignment. 

New dome control/mount control hardware can be easily integrated into the existing device-control node framework by writing new device drivers that implement this control interface.

### Dome controller:

- Figure out how to interact with this.
- Basic functionality:
    - Query current dome position (rotation) → how is this measured?
    - Command dome to rotate $-\pi \leq \theta \leq \pi$ degrees relative to an absolute ref (if it exists) or relative to its current position.
    - Command dome to open slit.

### Mount:

- Basic functionality:
    - Query current pointing information.