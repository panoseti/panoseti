#!/usr/bin/env python3

"""
The Python implementation of a gRPC DaqUtils client.
Requires the following to work:
    1. All Python packages specified in requirements.txt.
Run this on the headnode to configure the u-blox GNSS receivers in remote domes.
"""
import argparse
import logging
import sys
import signal
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
from daq_data_pb2 import PanoImage, StreamImagesResponse, StreamImagesRequest, InitHpIoRequest, InitHpIoResponse

## daq_data utils
from daq_data_resources import format_stream_images_response, make_rich_logger, unpack_pano_image, reflect_services
from daq_data_testing import run_all_tests, is_os_posix


def stream_images(
        stub: daq_data_pb2_grpc.DaqDataStub,
        stream_movie_data: bool,
        stream_pulse_height_data: bool,
        update_interval_seconds: float,
        logger: logging.Logger,
        wait_for_ready: bool = False,
):
    """Streams PanoImages from an active observing run."""
    # Create the request message
    stream_images_request = StreamImagesRequest(
        stream_movie_data=stream_movie_data,
        stream_pulse_height_data=stream_pulse_height_data,
        update_interval_seconds=update_interval_seconds,
    )
    # Make the RPC call
    stream_images_responses = stub.StreamImages(stream_images_request, wait_for_ready=wait_for_ready)
    try:
        # Process responses
        for stream_images_response in stream_images_responses:
            # optional: log response metadata
            formatted_stream_images_response = format_stream_images_response(stream_images_response)
            logger.info(formatted_stream_images_response)

            # Get pano images from response
            module_id, pano_type, header, image_array = unpack_pano_image(stream_images_response.pano_image)
            if pano_type == 'PULSE_HEIGHT':
                #
                # Your pulse-height visualizations here
                #
                ...
            elif pano_type == 'MOVIE':
                #
                # Your movie-mode visualizations here
                #
                ...
    finally:
        # Gracefully cancel RPC before exiting
        logger.info(f"'^C' received, closing connection to the DaqData server")
        if stream_images_responses is not None:
            stream_images_responses.cancel()

def init_hp_io(
        stub: daq_data_pb2_grpc.DaqDataStub,
        data_dir: str,
        update_interval_seconds: float,
        simulate_daq: bool,
        force: bool,
        module_ids: list[int],
        logger: logging.Logger,
        timeout:float=5.0,
) -> None:
    init_hp_io_request = InitHpIoRequest(
        data_dir=data_dir,
        update_interval_seconds=update_interval_seconds,
        simulate_daq=simulate_daq,
        force=force,
        module_ids=module_ids,
    )
    logger.info(f"Initializing the hp_io thread with "
                f"{MessageToDict(init_hp_io_request, preserving_proto_field_name=True, always_print_fields_with_no_presence=True)}")
    init_hp_io_response = stub.InitHpIo(init_hp_io_request, timeout=timeout)
    if init_hp_io_response.success:
        logger.info(f"init_hp_io_response={MessageToDict(init_hp_io_response, preserving_proto_field_name=True, always_print_fields_with_no_presence=True)}")
    else:
        logger.error(f"init_hp_io_response={MessageToDict(init_hp_io_response, preserving_proto_field_name=True, always_print_fields_with_no_presence=True)}")


def run(host, port=50051):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    logger = make_rich_logger(__name__, level=logging.INFO)

    # optional: run some client-side tests (e.g. check redis connection, check paths, etc)
    print("-------------- Client-side Tests --------------")
    all_pass, test_results = run_all_tests(
        test_fn_list=[
            is_os_posix
        ],
        args_list=[
            []
        ]
    )
    assert all_pass, "at least one client-side test failed"
    logger.info(f"all_pass={all_pass}")

    connection_target = f"{host}:{port}"
    try:
        with grpc.insecure_channel(connection_target) as channel:
            stub = daq_data_pb2_grpc.DaqDataStub(channel)
            print("-------------- ServerReflection --------------")
            reflect_services(channel)

            print("-------------- InitHpIo --------------")
            init_hp_io(
                stub,
                data_dir="/mnt/data10",
                update_interval_seconds=0.4,
                simulate_daq=True,
                force=True,
                timeout=10.0,
                logger=logger,
            )

            print("-------------- StreamImages --------------")
            stream_images(
                stub,
                stream_movie_data=True,
                stream_pulse_height_data=True,
                update_interval_seconds=1,
                wait_for_ready=True,
                logger=logger,
            )
    except KeyboardInterrupt:
        logger.info(f"'^C' received, closing connection to the DaqData server at {repr(connection_target)}")
    except grpc.RpcError as rpc_error:
        logger.error(f"{type(rpc_error)}\n{repr(rpc_error)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        help="daq_data server hostname or IP address. Default: 'localhost'",
        default="localhost"
    )
    args = parser.parse_args()
    run(host=args.host)
    # run(host="10.0.0.60")
