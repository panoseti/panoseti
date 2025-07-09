#!/usr/bin/env python3
import logging
import grpc
import daq_data_pb2
import daq_data_pb2_grpc
from daq_data_pb2 import PanoImage, StreamImagesResponse, StreamImagesRequest

from daq_data_client import reflect_services, unpack_pano_image, format_stream_images_response
from daq_data_resources import make_rich_logger
from daq_data_testing import run_all_tests, is_os_posix

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def show_stream_images(
        stub: daq_data_pb2_grpc.DaqDataStub,
        stream_movie_data: bool,
        stream_pulse_height_data: bool,
        update_interval_seconds: float,
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
        # Create plot
        fig, axs = plt.subplots(1, 2)
        plt.ion()  # Turn on interactive mode
        plt.show()
        # Randomness for demo
        cmap = np.random.choice(['magma', 'viridis', 'rocket', 'mako', 'flare_r'])
        ph_baseline = np.random.randint(100, 500)

        # Process responses
        for stream_images_response in stream_images_responses:
            # optional: log response metadata
            formatted_stream_images_response = format_stream_images_response(stream_images_response)
            logger.info(formatted_stream_images_response)

            # Get pano images from response
            pano_type, header, img = unpack_pano_image(stream_images_response.pano_image)
            if pano_type == 'PULSE_HEIGHT':
                img += ph_baseline
                high = np.quantile(img, 1.0)
                low = np.quantile(img, 0.05)
                axs[0].cla()
                axs[0].imshow(img, vmin=low, vmax=high, cmap=cmap)
                axs[0].set_title(f"{pano_type}\npkt_num={header['pkt_num']}")
            elif pano_type == 'MOVIE':
                high = np.quantile(img, 0.95)
                low = np.quantile(img, 0.05)
                axs[1].cla()
                axs[1].imshow(img, vmin=low, vmax=high, cmap=cmap)
                axs[1].set_title(f"{pano_type}\npkt_num={header['quabo_0']['pkt_num']}")
            plt.draw()
            plt.pause(0.1)
    finally:
        # Gracefully cancel RPC before exiting
        logger.info(f"'^C' received, closing connection to the DaqData server")
        if stream_images_responses is not None:
            stream_images_responses.cancel()


def run(host, port=50051):
    connection_target = f"{host}:{port}"
    try:
        with grpc.insecure_channel(connection_target) as channel:
            stub = daq_data_pb2_grpc.DaqDataStub(channel)
            print("-------------- ServerReflection --------------")
            reflect_services(channel)

            # print("-------------- Init --------------")
            # TODO: add InitHpIo

            print("-------------- StreamImages --------------")
            stream_movie_data = np.random.uniform() > 0.5
            stream_pulse_height_data = np.random.uniform() > 0.5
            if not stream_movie_data and not stream_pulse_height_data:
                stream_movie_data = True
                stream_pulse_height_data = True
            show_stream_images(
                stub,
                stream_movie_data,
                stream_pulse_height_data,
                update_interval_seconds=1,
                wait_for_ready=True
            )
    except KeyboardInterrupt:
        logger.info(f"'^C' received, closing connection to the DaqData server at {repr(connection_target)}")
    except grpc.RpcError as rpc_error:
        logger.error(f"{type(rpc_error)}\n{repr(rpc_error)}")


if __name__ == "__main__":
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
    # run(host="10.0.0.60")
    run(host="localhost")