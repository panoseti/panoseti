"""
Test cases for the InitF9t RPC.

These functions verify a specified F9t is properly configured.
Each function must have type Callable[..., Tuple[bool, str]] as the example below:

    def is_even(n: int) -> Tuple[bool, str]:
        if n % 2 == 0:
            return True, f"{n} is even"
        else:
            return False, f"{n} is odd"
"""
from ublox_control_resources import *
from pyubx2.ubxhelpers import cfgkey2name

from pyubx2 import POLL, UBX_PAYLOADS_POLL


def check_client_f9t_cfg_keys(required_f9t_cfg_keys: List[str], client_f9t_keys: List[str]) -> Tuple[bool, str]:
    """Verify the client's f9t config keys contain all required keys."""
    required_key_set = set(required_f9t_cfg_keys)
    client_key_set = set(client_f9t_keys)
    if set(required_key_set).issubset(client_key_set):
        return True, "all required f9t_cfg keys are present"
    else:
        required_key_diff = required_key_set.difference(client_key_set)
        return False, f"the given f9t_cfg is missing the following required keys: {required_key_diff}"


def is_device_valid(device: str) -> Tuple[bool, str]:
    if not device:
        return False, f"{device=} is empty"
    elif os.path.exists(device):
        return True, f"{device} is valid"
    else:
        return False, f"'{device}' does not exist"


def is_os_posix():
    """Verify the server is running in a POSIX environment."""
    if os.name == 'posix':
        return True, f"detected a POSIX-compliant system"
    else:
        return False, f"{os.name} is not supported yet"

def poll_nav_messages(send_queue) -> Tuple[bool, str]:
    count = 0
    test_msg = ""
    for nam in UBX_PAYLOADS_POLL:
        if nam[0:4] == "NAV-":
            test_msg += f"Polling {nam} message type..."
            msg = UBXMessage("NAV", nam, POLL)
            send_queue.put(msg)
            count += 1
    return True, f"sent {count} messages: " + test_msg


def check_f9t_dataflow(f9t_cfg):
    """
    Verify all packets specified in the 'packet_ids' fields of cfg are being received.
    NOTE: for now this is hardcoded for UBX packets.
    @return: True if all packets have been received, False otherwise.
    """
    device = f9t_cfg['device']
    timeout = f9t_cfg['timeout']
    # ubx_cfg = cfg['protocol']['ubx']
    packet_ids = [cfgkey2name(cfg_key) for cfg_key in f9t_cfg['cfg_key_settings'].keys()]
    print(packet_ids)

    # Initialize dict for recording whether we're receiving packets of each type.
    pkt_id_flags = {pkt_id: False for pkt_id in packet_ids}

    msg = ""
    try:

        with Serial(device, F9T_BAUDRATE, timeout=timeout) as stream:
            ubr = UBXReader(stream, protfilter=UBX_PROTOCOL)
            print('Verifying packets are being received... (If stuck at this step, re-run with the "init" option.)')

            for i in range(timeout):  # assumes config packets are send every second -> waits for timeout seconds.
                raw_data, parsed_data = ubr.read()  # blocking read operation -> waits for next UBX_PROTOCOL packet.
                if parsed_data:
                    for pkt_id in pkt_id_flags.keys():
                        if parsed_data.identity == pkt_id:
                            pkt_id_flags[pkt_id] = True
                if all(pkt_id_flags.values()):
                    print('All packets are being received.')
                    return True, msg
    except KeyboardInterrupt:
        print('Interrupted by KeyboardInterrupt.')
        return False, msg
    raise Exception(f'Not all packets are being received. Check the following for details: {pkt_id_flags=}')



