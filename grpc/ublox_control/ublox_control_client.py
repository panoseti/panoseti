#!/usr/bin/env python3

"""
The Python implementation of a gRPC UbloxControl client.
Requires the following to work:
    1. A valid network connection to the Redis database on the headnode with
    R/W user permissions to the Redis UBLOX hashset.
    2. All Python packages specified in requirements.txt.
Run this on the headnode to configure the u-blox GNSS receivers in remote domes.
"""
import logging
import queue
import random
import sys
import signal

import pyubx2
import redis
import re
import datetime

# rich formatting
from rich import print
from rich.pretty import pprint, Pretty
from rich.console import Console

## gRPC imports
import grpc

# gRPC reflection service: allows clients to discover available RPCs
from google.protobuf.descriptor_pool import DescriptorPool
from grpc_reflection.v1alpha.proto_reflection_descriptor_database import (
    ProtoReflectionDescriptorDatabase,
)
# Standard gRPC protobuf types
from google.protobuf.struct_pb2 import Struct
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf import timestamp_pb2

# protoc-generated marshalling / demarshalling code
import ublox_control_pb2
import ublox_control_pb2_grpc

# alias messages to improve readability
from ublox_control_pb2 import CaptureUbloxRequest, InitF9tResponse, InitF9tRequest, CaptureUbloxResponse


## our code
from ublox_control_resources import *

# Gracefully cancel active RPCs before exiting
active_calls = []
def cancel_requests(unused_signum, unused_frame):
    """Signal handler to cancel all in-flight gRPCs."""
    for future in active_calls:
        future.cancel()
    sys.exit(0)
signal.signal(signal.SIGINT, cancel_requests)


def get_services(channel):
    """Prints all available RPCs for the UbloxControl service represented by [channel]."""
    def format_rpc_service(method):
        name = method.name
        input_type = method.input_type.name
        output_type = method.output_type.name
        client_stream = "stream " if method.client_streaming else ""
        server_stream = "stream " if method.server_streaming else ""
        return f"rpc {name}({client_stream}{input_type}) returns ({server_stream}{output_type})"
    reflection_db = ProtoReflectionDescriptorDatabase(channel)
    services = reflection_db.get_services()
    print(f"found services: {services}")

    desc_pool = DescriptorPool(reflection_db)
    service_desc = desc_pool.FindServiceByName("ubloxcontrol.UbloxControl")
    print(f"found UbloxControl service with name: {service_desc.full_name}")
    for method in service_desc.methods:
        print(f"\tfound: {format_rpc_service(method)}")


def init_f9t(stub, f9t_cfg, timeout=10) -> dict:
    """Initializes an F9T device according to the specification in init_f9t_request."""
    init_f9t_request = InitF9tRequest(f9t_cfg=ParseDict(f9t_cfg, Struct()))
    init_f9t_response_future = stub.InitF9t.future(init_f9t_request, timeout=timeout)
    active_calls.append(init_f9t_response_future)
    init_f9t_response = init_f9t_response_future.result()
    active_calls.pop()

    # unpack init_f9t_response
    init_status = InitF9tResponse.InitStatus.Name(init_f9t_response.init_status)
    curr_f9t_cfg = MessageToDict(init_f9t_response.f9t_cfg, preserving_proto_field_name=True)
    print("init_f9t_response.f9t_cfg=", end='')
    pprint(curr_f9t_cfg, expand_all=True)
    print(init_f9t_response.test_results)
    print(f'{init_f9t_response.message=}')
    print(f'init_f9t_response.status=', init_status)
    # for i, test_result in enumerate(init_f9t_response.test_results):
    #     print(f'TEST {i}:')
    #     print("\t" + str(test_result).replace("\n", "\n\t"))
    return curr_f9t_cfg


def capture_ublox(stub, patterns, f9t_cfg, timeout=10):
    # valid_capture_command_aliases = ['start', 'stop']

    def make_capture_ublox_request(pats):
        if pats is None:
            return CaptureUbloxRequest()
        for pat in pats:
            re.compile(pat)  # verify each regex pattern compiles
        return CaptureUbloxRequest(patterns=pats)

    def format_gnss_packet(packet_type, name, message, parsed_data, timestamp: datetime.datetime):
        timestamp = timestamp.isoformat()
        packet_type = CaptureUbloxResponse.Type.Name(packet_type)
        return f"{name=}, {parsed_data}, {message=}, {timestamp=}, {packet_type=}"

    def write_gnss_packet_to_redis(r, chip_name, chip_uid, packet_id, parsed_data, timestamp: datetime.datetime):
        # curr_time = datetime.datetime.now(tz=datetime.timezone.utc)
        # TODO: write methods to unpack parsed data
        rkey = get_f9t_redis_key(chip_name, chip_uid, packet_id)
        for k, v in parsed_data.items():
            r.hset(rkey, k, v)
        timestamp_float = timestamp.timestamp()  # timestamp when packet was received by the daq node
        r.hset(rkey, 'Computer_UTC', timestamp_float)

    # start packet stream
    capture_ublox_response_future = stub.CaptureUblox(
        make_capture_ublox_request(patterns)
    )
    # use active_calls to gracefully handle ^C cancellation
    active_calls.append(capture_ublox_response_future)

    # write stream return from the server
    redis_host, redis_port = "localhost", 6379
    chip_name = f9t_cfg['chip_name']
    chip_uid = f9t_cfg['chip_uid']
    with redis.Redis(host=redis_host, port=redis_port) as r:
        for gnss_packet in capture_ublox_response_future:
            packet_type = gnss_packet.type
            packet_id = gnss_packet.name
            message = gnss_packet.message
            parsed_data = MessageToDict(gnss_packet.parsed_data)
            timestamp = gnss_packet.timestamp.ToDatetime()
            if packet_type == CaptureUbloxResponse.Type.DATA:
                logger.debug(f"CaptureUbloxResponse: {format_gnss_packet(packet_type, packet_id, message, parsed_data, timestamp)}")
                write_gnss_packet_to_redis(r, chip_name, chip_uid, packet_id, parsed_data, timestamp)
            elif packet_type == CaptureUbloxResponse.Type.ERROR:
                if packet_id == "UNEXPECTED_F9T_DISCONNECTION":
                    logger.critical(f"CaptureUbloxResponse: {format_gnss_packet(packet_type, packet_id, message, parsed_data, timestamp)}")
                elif packet_id == "INVALID_SERVER_STATE":
                    logger.warning(f"CaptureUbloxResponse: {format_gnss_packet(packet_type, packet_id, message, parsed_data, timestamp)}")
                else:
                    logger.error(f"CaptureUbloxResponse: {format_gnss_packet(packet_type, packet_id, message, parsed_data, timestamp)}")


def run(host, port=50051):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.

    connection_target = f"{host}:{port}"
    try:
        with grpc.insecure_channel(connection_target) as channel:
            stub = ublox_control_pb2_grpc.UbloxControlStub(channel)

            print("-------------- ServerReflection --------------")
            get_services(channel)


            #for i in range(1):
            print("-------------- InitF9t --------------")
            client_f9t_cfg = default_f9t_cfg
            client_f9t_cfg['chip_uid'] = 'BEEFEDDEAD'

            curr_f9t_cfg = client_f9t_cfg
            # curr_f9t_cfg = init_f9t(stub, client_f9t_cfg, 5)

            print("-------------- CaptureUblox --------------")
            capture_ublox(stub, None, curr_f9t_cfg, 5)
    except KeyboardInterrupt:
        logger.info(f"'^C' received, closing connection to UbloxControl server at {repr(connection_target)}")
    except grpc.RpcError as rpc_error:
        logger.error(f"{type(rpc_error)}\n{repr(rpc_error)}")


if __name__ == "__main__":
    # logging.basicConfig()
    logger = make_rich_logger(__name__)

    # Run client-side tests
    print("-------------- Client-side Tests --------------")
    all_pass, _ = run_all_tests(
        test_fn_list=[
            test_redis_connection,
        ],
        args_list=[
            ["localhost", 6379, 1, logger]
        ]
    )
    assert all_pass, "at least one client-side test failed"
    # test_redis_connection("localhost", logger=logger)
    # run(host="10.0.0.60")
    run(host="localhost")


