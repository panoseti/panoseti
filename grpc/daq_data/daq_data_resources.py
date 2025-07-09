"""
Common functions for the DaqData clients and servers
"""
import sys, signal
import logging
from typing import List, Callable, Tuple, Any, Dict
import numpy as np
from pandas import to_datetime, Timestamp
import datetime
import decimal

# rich formatting
from rich import print
from rich.logging import RichHandler
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
from daq_data_pb2 import TestCase, PanoImage, StreamImagesResponse, StreamImagesRequest

# panoseti utils
sys.path.append('../../util')
import pff


def make_rich_logger(name, level=logging.INFO):
    LOG_FORMAT = (
        "[tid=%(thread)d] [%(funcName)s()] %(message)s "
    )

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
        # datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    return logging.getLogger(name)


def reflect_services(channel: grpc.Channel) -> None:
    """Prints all available RPCs for a DaqData service represented by [channel]."""
    def format_rpc_service(method):
        name = method.name
        input_type = method.input_type.name
        output_type = method.output_type.name
        stream_fmt = '[magenta]stream[/magenta] '
        client_stream = stream_fmt if method.client_streaming else ""
        server_stream = stream_fmt if method.server_streaming else ""
        return f"rpc {name}({client_stream}{input_type}) returns ({server_stream}{output_type})"
    reflection_db = ProtoReflectionDescriptorDatabase(channel)
    services = reflection_db.get_services()
    print(f"found services: {services}")

    desc_pool = DescriptorPool(reflection_db)
    service_desc = desc_pool.FindServiceByName("daqdata.DaqData")
    print(f"found [yellow]DaqData[/yellow] service with name: [yellow]{service_desc.full_name}[/yellow]")
    for method in service_desc.methods:
        print(f"\tfound: {format_rpc_service(method)}")

def parse_pano_timestamps(pano_image: daq_data_pb2.PanoImage) -> Dict[str, Any]:
    """Parse PanoImage header to get nanosecond-precision timestamps."""
    h = MessageToDict(pano_image.header)
    td = {}
    # Add nanosecond-precision Pandas Timestamp from panoseti packet timing
    if pano_image.shape == [16, 16]:
        td['wr_unix_timestamp'] = pff.wr_to_unix_decimal(h['pkt_tai'], h['pkt_nsec'], h['tv_sec'])
    elif pano_image.shape == [32, 32]:
        h_q0 = h['quabo_0']
        td['wr_unix_timestamp'] = pff.wr_to_unix_decimal(h_q0['pkt_tai'], h_q0['pkt_nsec'], h_q0['tv_sec'])
    nanoseconds_since_epoch = int(td['wr_unix_timestamp'] * decimal.Decimal('1e9'))
    td['pandas_unix_timestamp'] = to_datetime(nanoseconds_since_epoch, unit='ns')
    return td

def unpack_pano_image(
        pano_image: daq_data_pb2.PanoImage
) -> Tuple[str, Dict[str, Any], np.ndarray] | Tuple[None, None, None]:
    """Unpacks a PanoImage message into its components."""
    if pano_image is None:
        return None, None, None
    pano_type = PanoImage.Type.Name(pano_image.type)
    # Parse header
    h = MessageToDict(pano_image.header)
    pano_timestamps = parse_pano_timestamps(pano_image)
    h.update(pano_timestamps)

    image_array = np.array(pano_image.image_array).reshape(pano_image.shape)
    bytes_per_pixel = pano_image.bytes_per_pixel
    if bytes_per_pixel == 1:
        image_array = image_array.astype(np.uint8)
    elif bytes_per_pixel == 2:
        image_array = image_array.astype(np.uint16)
    else:
        raise ValueError(f"unsupported bytes_per_pixel: {bytes_per_pixel}")
    return pano_type, h, image_array

def format_stream_images_response(stream_images_response: StreamImagesResponse) -> str:
    pano_type, header, image_array = unpack_pano_image(stream_images_response.pano_image)
    name = stream_images_response.name
    message = stream_images_response.message
    server_timestamp = stream_images_response.timestamp.ToDatetime().isoformat()
    return f"StreamImagesResponse: {name=}, {message=}, {server_timestamp=}, {header=}, {pano_type=}"

# Gracefully cancel active RPCs before exiting
# active_calls = []
# def cancel_requests(unused_signum, unused_frame):
#     """Signal handler to cancel all in-flight gRPCs."""
#     for future in active_calls:
#         future.cancel()
#     sys.exit(0)
# signal.signal(signal.SIGINT, cancel_requests)

