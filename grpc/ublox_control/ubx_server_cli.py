#!/usr/bin/env python3

"""
Command Line Interface for the Server-side RPC methods.
These commands expect the following to function correctly:
    1. A valid network connection to the Redis database on the headnode.
    2. R/W user permissions to the Redis UBLOX hashset.
    3. A valid ZED-F9T u-blox chip is available as a /dev/... file.
    4. All required Python packages are installed.

See https://github.com/semuconsulting/pyubx2 for documentation on UBX interface documentation.
"""

import time
import os
import datetime
import argparse


from ublox_control_resources import *

packet_data_dir = 'data'

"""Server utility functions"""


""" Initialize u-blox device. """
def init(args):
    """Configure device and verify all desired packets are being received."""
    device = args.device
    check_device_exists(device)
    poll_f9t_config(device)
    set_f9t_config(device)
    poll_f9t_config(device)
    verified = check_f9t_dataflow(device)     # Thows an Exception if not all packet types are being received.
    if not verified:
        return False
    print(f"Device initialized. Ready to collect data!")
    f9t_config["init_success"] = True
    return True



def test_redis_cli_connection(args):
    """
    Server-side (on DAQ node) cli for RPC methods. Intended for testing.
    Returns True iff all test cases passed.
    """

    host = args.host
    port = args.port
    socket_timeout = args.timeout

    start_timestamp = datetime.datetime.now().isoformat()
    print('Starting redis test:'
          '\n\tUse CTRL+C to stop.'
          '\n\tRun "HGETALL TEST" to view test keys on the Redis server'
          '\n\tStart timestamp: {}\n'.format(start_timestamp))
    failures = 1
    try:
        failures = test_redis_connection(host, port, socket_timeout)
    except KeyboardInterrupt:
        print(f'\nStopping test. {failures=}')
        return False
    return failures == 0


""" Collect packets and forward to Redis. """
def collect_data(r: redis.Redis, device: str, cfg=f9t_config):
    timeout = cfg['timeout (s)']
    ubx_cfg = cfg['protocol']['ubx']

    # Cache for saving packets and timestamping their unix arrival time using host computer clock.
    packet_cache = {}
    for pkt_id in ubx_cfg['packet_ids']:
        packet_cache[pkt_id] = {'valid': False, 'parsed_data': None}

    def all_packets_valid():
        all_valid = True
        for pkt_id in ubx_cfg['packet_ids']:
            all_valid &= packet_cache[pkt_id]['valid']
        return all_valid

    def flush_packet_cache():
        curr_time = time.time()  # datetime.datetime.now()
        chip_name = cfg['chip_name']
        chip_uid = cfg['chip_uid']
        # Pipeline Redis key updates for efficiency.
        pipe = r.pipeline()
        for pkt_id in ubx_cfg['packet_ids']:
            prot_msg = f"UBX-{pkt_id}"  # # just ubx packets for now
            rkey = get_f9t_redis_key(chip_name, chip_uid, prot_msg)
            for k, v in packet_cache[pkt_id].items():
                pipe.hset(rkey, k, v)
            pipe.hset(rkey, 'Computer_UTC', curr_time)
            packet_cache[pkt_id]['valid'] = False # invalidate packet cache entry
        pipe.execute()

    with (Serial(device, F9T_BAUDRATE, timeout=timeout) as stream):
        ubr = UBXReader(stream, protfilter=UBX_PROTOCOL)
        while True:
            # Wait for next packet (blocking read)
            raw_data, parsed_data = ubr.read()

            # Add parsed data to cache
            if parsed_data is not None:
                pkt_id_curr = parsed_data.identity
                # Add new packet to packet cache
                if pkt_id_curr in packet_cache:
                    # First flush cache to Redis if the current packet will overwrite cached data
                    if packet_cache[pkt_id_curr]['valid']:
                        flush_packet_cache()
                    packet_cache[pkt_id_curr]['valid'] = True
                    packet_cache[pkt_id_curr]['parsed_data'] = parsed_data.to_dict()

            # Flush cache to Redis if all u-blox hk were received
            if all_packets_valid():
                flush_packet_cache()

def start_collect(args):
    device = args.device
    valid_dataflow = check_f9t_dataflow(device)     # Will throw Exception if not all packet types are being received.
    if not valid_dataflow:
        return False
    # Connect to Redis database
    r = redis.Redis(
        host="localhost", # TODO
        port=6379 # TODO
    )

    # Start data collection
    start_timestamp = datetime.datetime.now().isoformat()
    # experiment_dir = get_experiment_dir(start_timestamp, device)
    # os.makedirs(experiment_dir, exist_ok=False)
    print('Starting data collection. To stop collection, use CTRL+C.'
          '\nStart timestamp: {}'.format(start_timestamp))
    try:
        collect_data(r, device)
    except KeyboardInterrupt:
        print('Stopping data collection.')
        pass
    # finally:
        # Save data
        # for data_type in df_refs.keys():
        #     fpath = f'{experiment_dir}/data-type_{data_type}.start_{start_timestamp}'
        #     save_data(df_refs[data_type], fpath)
        # print('Data saved in {}'.format(experiment_dir))




def cli_handler():
    # create the top-level parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    # init command parser
    parser_init = subparsers.add_parser('init',
                                        description='Configures u-blox device to start sending the specified packets and verifies they are all being received.')
    parser_init.add_argument('device',
                             help='specify the device file path. e.g. /dev/ttyS3',
                             type=str)
    parser_init.set_defaults(func=init)

    # collect command parser
    parser_collect = subparsers.add_parser('collect',
                                           description='Start data collection. (To stop collection, use CTRL+C.)')
    parser_collect.add_argument('device',
                                help='specify the device path. example: /dev/ttyS3',
                                type=str)
    parser_collect.set_defaults(func=start_collect)

    # test command parser

    parser_test_redis = subparsers.add_parser('test_redis', description='Test Redis connection')
    parser_test_redis.add_argument('host',
                                help='host or ip address of the computer running the Redis server',
                                type=str,
                                )
    parser_test_redis.add_argument('port',
                                help='port of the Redis server. Default: 6379',
                                nargs="?",
                                type=int,
                                default=6379)
    parser_test_redis.add_argument('timeout',
                                   help='cli connection timeout seconds. Default: 5',
                                   nargs="?",
                                   type=int,
                                   default=5)
    # parser_test_redis.add_argument("-n", "--interval_seconds",
    #                                # action="store_true",
    #                                help="number of seconds between test write operations. Default: 1 second",
    #                                type=int,
    #                                default=1
    #                                )
    parser_test_redis.set_defaults(func=test_redis_cli_connection)

    # Dispatch command action
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    cli_handler()
