#!/usr/bin/env python3

import sys
import argparse
import logging
from collections import deque
import time

import grpc
import daq_data_pb2
import daq_data_pb2_grpc
from daq_data_pb2 import PanoImage, StreamImagesResponse, StreamImagesRequest

from daq_data_client import reflect_services, unpack_pano_image, format_stream_images_response, init_hp_io
from daq_data_resources import make_rich_logger
from daq_data_testing import run_all_tests, is_os_posix

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

class PulseHeightDistribution:
    def __init__(self, durations_seconds):
        self.durations = durations_seconds
        self.start_times = [time.time() for _ in durations_seconds]
        self.hist_data = [deque() for _ in durations_seconds]

        # Set reasonable size: width=6in, height=3in per subplot
        n = len(durations_seconds)
        height = max(3 * n, 6)  # Ensure minimum height
        plt.ion()
        self.fig, self.axes = plt.subplots(n, 1, figsize=(6, height))
        if n == 1:
            self.axes = [self.axes]

    def update(self, image):
        max_pixel = int(np.max(image))
        now = time.time()
        for i, duration in enumerate(self.durations):
            if now - self.start_times[i] > duration:
                self.hist_data[i].clear()
                self.start_times[i] = now
            self.hist_data[i].append(max_pixel)

    def plot(self):
        palette = sns.color_palette('husl', len(self.durations))
        for i, (duration, values) in enumerate(zip(self.durations, self.hist_data)):
            ax = self.axes[i]
            ax.clear()
            if values:
                sns.histplot(
                    list(values),
                    bins=100,
                    kde=False,
                    stat='density',
                    element='step',
                    label=f'{duration}s',
                    color=palette[i],
                    ax=ax,
                )
            ax.set_title(f"Pulse-Height Distribution: {duration}s")
            ax.set_xlabel("ADC Value")
            ax.set_ylabel("Density")
            ax.legend(title="Duration")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class PanoImagePreviewer:
    def __init__(self, stream_movie_data: bool, stream_pulse_height_data: bool, update_interval_seconds: float, logger: logging.Logger):
        self.stream_movie_data = stream_movie_data
        self.stream_pulse_height_data = stream_pulse_height_data
        self.update_interval_seconds = update_interval_seconds
        self.logger = logger

        # Initialize plotting - two subplots side-by-side
        self.fig, self.axs = plt.subplots(1, 2)
        for i, ax in enumerate(self.axs):
            ax.imshow(np.zeros((32, 32)))
            if i == 0 and not stream_pulse_height_data:
                ax.set_title(f'stream_pulse_height_data={stream_pulse_height_data}')
            elif i == 1 and not stream_movie_data:
                ax.set_title(f'stream_movie_data={stream_movie_data}')
            ax.axis('off')
        plt.ion()  # Enable interactive mode
        plt.show()

        # Randomly choose a color map from a set of options
        self.cmap = np.random.choice(['magma', 'viridis', 'rocket', 'mako', 'icefire', 'flare_r'])
        self.ph_baseline = 700
        self.text_width = 35
        self.font_size = 9

        # Buffers to store images for quantile computations (max size 100)
        self.movie_imgs = []
        self.ph_imgs = []

    def update(self, pano_image_response):
        # Log response metadata for diagnostics
        formatted_response = format_stream_images_response(pano_image_response)
        self.logger.info(formatted_response)

        # Extract pano image data
        pano_image = pano_image_response.pano_image
        pano_type, header, img = unpack_pano_image(pano_image)

        # Update figure title with the image acquisition date
        plt_title = f"demo obs data from {header['pandas_unix_timestamp'].date()}"
        self.fig.suptitle(plt_title)

        # Compose axis title with metadata like time, frame number, and source file information
        ax_title = (f"{pano_type}\n"
                    f"unix_t = {header['pandas_unix_timestamp'].time()}\n"
                    f"frame_no = {pano_image.frame_number}\n")
        ax_title += textwrap.fill(f"file = {pano_image.file}", width=self.text_width)

        # Update pulse height image subplot
        if pano_type == 'PULSE_HEIGHT':
            if len(self.ph_imgs) < 100:
                self.ph_imgs.append(img)
            img += self.ph_baseline
            img = np.clip(img, self.ph_baseline, float('inf'))
            img -= self.ph_baseline
            high = np.quantile(self.ph_imgs, 0.99)
            self.axs[0].cla()
            self.axs[0].imshow(img, vmin=0, vmax=high, cmap=self.cmap)
            self.axs[0].set_title(ax_title, fontsize=self.font_size)

        # Update movie image subplot
        elif pano_type == 'MOVIE':
            if len(self.movie_imgs) < 100:
                self.movie_imgs.append(img)
            high = np.quantile(img, 0.95)
            low = np.quantile(img, 0.05)
            self.axs[1].cla()
            self.axs[1].imshow(img, vmin=low, vmax=high, cmap=self.cmap)
            self.axs[1].set_title(ax_title, fontsize=self.font_size)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def run_max_pixel_distribution_ph(
    stub,
    plot_update_interval: float,
    durations_seconds=(5, 10, 30),
    logger: logging.Logger = None,
):
    """Streams pulse-height images and updates max pixel distribution histograms."""
    # Build the request for pulse-height image streaming only
    request = StreamImagesRequest(
        stream_movie_data=False,
        stream_pulse_height_data=True,
        update_interval_seconds=-1
    )
    stream_images_responses = stub.StreamImages(request)

    mpd = PulseHeightDistribution(durations_seconds)
    last_plot_update_time = time.time()
    for response in stream_images_responses:
        # log response metadata
        if logger:
            formatted_stream_images_response = format_stream_images_response(response)
            logger.info(formatted_stream_images_response)

        # unpack pano image
        pano_image = response.pano_image
        pano_type, header, img = unpack_pano_image(pano_image)
        ph_baseline = 700

        if pano_type == 'PULSE_HEIGHT':
            img += ph_baseline
            img = np.clip(img, ph_baseline, float('inf'))
            img -= ph_baseline
            mpd.update(img)
            curr_time = time.time()
            if curr_time - last_plot_update_time > max(plot_update_interval, 0.5):
                mpd.plot()
                last_plot_update_time = curr_time


def preview_data_demo(
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
    previewer = PanoImagePreviewer(stream_movie_data, stream_pulse_height_data, update_interval_seconds, logger)

    # Process responses
    for stream_images_response in stream_images_responses:
        # log response metadata
        formatted_stream_images_response = format_stream_images_response(stream_images_response)
        logger.info(formatted_stream_images_response)
        previewer.update(stream_images_response)


def run(host, port=50051, init=False, simulate_daq=False, plot='prev'):
    logger = make_rich_logger(__name__, level=logging.INFO)
    connection_target = f"{host}:{port}"
    logger.info(f"connection_target={repr(connection_target)}")
    try:
        with grpc.insecure_channel(connection_target) as channel:
            stub = daq_data_pb2_grpc.DaqDataStub(channel)
            print("-------------- ServerReflection --------------")
            reflect_services(channel)

            # print("-------------- Init --------------")
            if init:
                init_hp_io(
                    stub,
                    data_dir="/mnt/data10",
                    update_interval_seconds=0.1,
                    simulate_daq=simulate_daq,
                    force=True,
                    timeout=15.0,
                    logger=logger
                )

            print("-------------- StreamImages --------------")
            if plot == 'prev':
                preview_data_demo(
                    stub,
                    stream_movie_data=True,
                    stream_pulse_height_data=True,
                    update_interval_seconds=0.5,
                    wait_for_ready=True,
                    logger=logger
                )
            elif plot == 'phdist':
                run_max_pixel_distribution_ph(
                    stub,
                    plot_update_interval=0.5,
                    durations_seconds= (10, 60, 60 * 10),
                    logger=logger
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
    parser.add_argument(
        "--init",
        help="initialize an hp_io thread to track an active run directory",
        action="store_true"
    )
    parser.add_argument(
        "--sim",
        help="use a simulated datastream",
        action="store_true"
    )

    parser.add_argument(
        "--plot",
        help="use a simulated datastream",
        choices=['prev', 'phdist'],
        default='prev'
    )
    # run(host="10.0.0.60")
    args = parser.parse_args()
    run(host=args.host, init=args.init, simulate_daq=args.sim, plot=args.plot)
