#!/usr/bin/env python3

"""
The Python implementation of a gRPC DaqUtils client.
Requires the following to work:
    1. All Python packages specified in requirements.txt.
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
import numpy as np

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
import daq_data_pb2
import daq_data_pb2_grpc
from daq_data_pb2 import TestCase, StreamImagesResponse, StreamImagesRequest

## our code
from daq_data_resources import *
from daq_data_testing import *

# Gracefully cancel active RPCs before exiting
active_calls = []
def cancel_requests(unused_signum, unused_frame):
    """Signal handler to cancel all in-flight gRPCs."""
    for future in active_calls:
        future.cancel()
    sys.exit(0)
signal.signal(signal.SIGINT, cancel_requests)


def reflect_services(channel):
    """Prints all available RPCs for the DaqData service represented by [channel]."""
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
    service_desc = desc_pool.FindServiceByName("daqdata.DaqData")
    print(f"found DaqData service with name: {service_desc.full_name}")
    for method in service_desc.methods:
        print(f"\tfound: {format_rpc_service(method)}")

def make_stream_images_request(stream_movie_data, stream_pulse_height_data):
    return StreamImagesRequest(
        stream_movie_data=stream_movie_data,
        stream_pulse_height_data=stream_pulse_height_data,
    )

def unpack_pano_image(pano_image) -> Tuple[Dict, np.ndarray]:
    if pano_image is None:
        return None, None
    image_shape = pano_image.image_shape
    bytes_per_pixel = pano_image.bytes_per_pixel
    image_array = np.array(pano_image.image_array).reshape(image_shape)
    if bytes_per_pixel == 1:
        image_array = image_array.astype(np.uint8)
    elif bytes_per_pixel == 2:
        image_array = image_array.astype(np.uint16)
    else:
        raise ValueError(f"unsupported bytes_per_pixel: {bytes_per_pixel}")
    header = MessageToDict(pano_image.header)
    return header, image_array

def format_stream_images_response(stream_images_response):
    header, image_array = unpack_pano_image(stream_images_response.pano_image)
    name = stream_images_response.name
    message = stream_images_response.message
    timestamp = stream_images_response.timestamp.ToDatetime().isoformat()
    return f"StreamImagesResponse: {name=}, {message=}, {timestamp=}, {header=}"


def stream_images(stub, stream_movie_data, stream_pulse_height_data, timeout=10):
    logger = make_rich_logger(__name__, level=logging.INFO)

    # start packet stream
    stream_images_request = make_stream_images_request(stream_movie_data, stream_pulse_height_data)
    stream_images_responses = stub.StreamImages(stream_images_request)
    active_calls.append(stream_images_responses)  # gracefully handle ^C cancellation

    import matplotlib.pyplot as plt
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()

    for stream_images_response in stream_images_responses:
        # display a log message
        formatted_stream_images_response = format_stream_images_response(stream_images_response)
        logger.info(formatted_stream_images_response)

        # simple image display
        header, img = unpack_pano_image(stream_images_response.pano_image)

        ax.imshow(img)
        plt.draw()
        plt.pause(0.5)  # Pause to simulate "live" streaming
        ax.clear()

        # plt.ioff()
        plt.show()


def run(host, port=50051):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.

    connection_target = f"{host}:{port}"
    try:
        with grpc.insecure_channel(connection_target) as channel:
            print("-------------- ServerReflection --------------")
            reflect_services(channel)

            stub = daq_data_pb2_grpc.DaqDataStub(channel)

            # TODO: add InitHpIo
            # print("-------------- Init --------------")
            # client_hashpipe_io_cfg = default_hp_io_thread_config
            # curr_f9t_cfg = client_hashpipe_io_cfg

            print("-------------- StreamImages --------------")
            stream_images(stub, True, True, 5)

    except KeyboardInterrupt:
        logger.info(f"'^C' received, closing connection to the DaqData server at {repr(connection_target)}")
    except grpc.RpcError as rpc_error:
        logger.error(f"{type(rpc_error)}\n{repr(rpc_error)}")


if __name__ == "__main__":
    # logging.basicConfig()
    logger = make_rich_logger(__name__, level=logging.INFO)

    # Run client-side tests
    print("-------------- Client-side Tests --------------")
    all_pass, _ = run_all_tests(
        test_fn_list=[
        ],
        args_list=[
        ]
    )
    assert all_pass, "at least one client-side test failed"
    # test_redis_connection("localhost", logger=logger)
    # run(host="10.0.0.60")
    run(host="localhost")


