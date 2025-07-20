#!/usr/bin/env python3

import signal
import argparse
import json
import logging
import os.path
from collections import deque
import time
import sys

import grpc
from google.protobuf.json_format import MessageToDict

import daq_data_pb2
import daq_data_pb2_grpc
from daq_data_pb2 import PanoImage, StreamImagesResponse, StreamImagesRequest

from daq_data_client import reflect_services, unpack_pano_image, format_stream_images_response, init_hp_io
from daq_data_resources import make_rich_logger

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

class PulseHeightDistribution:
    VMIN = 0
    VMAX = 2**12 - 1  # 4095
    def __init__(self, durations_seconds, module_ids):
        self.durations = durations_seconds
        self.module_ids = module_ids
        n = len(durations_seconds)  # num of plots to make
        self.start_times = [time.time() for _ in range(n)]
        self.hist_data = [deque() for _ in range(n)]
        self.vmins = [self.VMAX for _ in range(n)]
        self.vmaxs = [self.VMIN for _ in range(n)]
        # size: width=6in, height=3in per subplot
        height = max(2.9 * n, 6)  # ensure a minimum height
        plt.ion()
        self.fig, self.axes = plt.subplots(n, 1, figsize=(6, height))
        self.fig.suptitle(f'Pulse Height Distributions for {module_ids=}')
        if n == 1:
            self.axes = [self.axes]

    def update(self, image):
        max_pixel = int(np.max(image))
        now = time.time()
        for i, duration in enumerate(self.durations):
            if now - self.start_times[i] > duration:
                self.hist_data[i].clear()
                self.start_times[i] = now
                self.vmins[i] = self.VMAX
                self.vmaxs[i] = self.VMIN
            self.hist_data[i].append(max_pixel)
            self.vmins[i] = min(self.vmins[i], max_pixel)
            self.vmaxs[i] = max(self.vmaxs[i], max_pixel)

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
            ax.set_xlim(min(self.vmins) - 10, max(self.vmaxs) + 10)
            ax.set_title(f"Pulse-Height Distribution: {duration}s")
            ax.set_xlabel("ADC Value")
            ax.set_ylabel("Density")
            ax.legend(title="Duration")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class PanoImagePreviewer:
    def __init__(
            self,
            stream_movie_data: bool,
            stream_pulse_height_data: bool,
            update_interval_seconds: float,
            logger: logging.Logger,
            text_width=25,
            font_size=7,
            ph_baseline = 750,
            col_width=4,
            row_height=2.8,
    ) -> None:
        self.stream_movie_data = stream_movie_data
        self.stream_pulse_height_data = stream_pulse_height_data
        self.update_interval_seconds = update_interval_seconds
        self.logger = logger

        self.seen_modules = set()
        self.axes_map = {}

        self.fig = None
        self.text_width = text_width
        self.font_size = font_size
        self.ph_baseline = ph_baseline
        self.cmap = np.random.choice(['magma', 'viridis', 'rocket', 'mako', 'icefire', 'flare_r'])
        self.col_width = col_width
        self.row_height = row_height

    def safe_imshow(self, ax, img, vmin, vmax, cmap):
        """Ensure vmin < vmax for Matplotlib imshow."""
        if vmin >= vmax:
            vmax = vmin + 1e-6
        ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)

    def setup_layout(self, modules):
        """Sets up subplot layout: one row per module, two columns (PH left, Movie right)."""
        if self.fig is not None:
            plt.close(self.fig)
        modules = sorted(modules)
        n_modules = len(modules)
        self.fig, axs = plt.subplots(n_modules, 2, figsize=(self.col_width, self.row_height * n_modules))
        if n_modules == 1:
            axs = np.array([axs])  # one row per module

        self.axes_map.clear()
        for row, mod_id in enumerate(modules):
            self.axes_map[(mod_id, 'PULSE_HEIGHT')] = axs[row, 0]
            self.axes_map[(mod_id, 'MOVIE')] = axs[row, 1]
            axs[row, 0].imshow(np.zeros((32, 32)))
            axs[row, 1].imshow(np.zeros((32, 32)))
            axs[row, 0].set_title(f'Module {mod_id} - Pulse-Height', fontsize=self.font_size)
            axs[row, 1].set_title(f'Module {mod_id} - Movie-Mode', fontsize=self.font_size)
            axs[row, 0].axis('off')
            axs[row, 1].axis('off')
        self.fig.tight_layout()
        plt.ion()
        plt.show()

    def update(self, pano_image, pano_type, header, img, module_id):

        if module_id not in self.seen_modules:
            self.seen_modules.add(module_id)
            self.setup_layout(self.seen_modules)

        ax = self.axes_map.get((module_id, pano_type))
        if ax is None:
            return

        # Prepare axis title with details
        ax_title = (f"{pano_type}"
                    + ("\n" if 'quabo_num' not in header else f": Q{int(header['quabo_num'])}\n")
                    + f"unix_t = {header['pandas_unix_timestamp'].time()}\n"
                    + f"frame_no = {pano_image.frame_number}\n")
        ax_title += textwrap.fill(f"file = {pano_image.file}", width=self.text_width)

        if pano_type == 'PULSE_HEIGHT':
            img_mod = img + self.ph_baseline
            vmin = self.ph_baseline
            vmax = np.quantile(img_mod, 0.99)
            ax.cla()
            self.safe_imshow(ax, img_mod, vmin, vmax, self.cmap)
        elif pano_type == 'MOVIE':
            vmin = np.quantile(img, 0.05)
            vmax = np.quantile(img, 0.95)
            ax.cla()
            self.safe_imshow(ax, img, vmin, vmax, self.cmap)

        ax.set_title(ax_title, fontsize=self.font_size)
        # ax.axis('off')

        plt_title = f"Obs data from {header['pandas_unix_timestamp'].date()}"
        self.fig.suptitle(plt_title)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def run_pulse_height_distribution(
    stub,
    plot_update_interval: float,
    module_ids: int,
    durations_seconds=(5, 10, 30),
    logger: logging.Logger = None,
):
    """Streams pulse-height images and updates max pixel distribution histograms."""
    # Build the request for pulse-height image streaming only
    stream_images_request = StreamImagesRequest(
        stream_movie_data=False,
        stream_pulse_height_data=True,
        update_interval_seconds=-1,
        module_ids=module_ids,
    )
    logger.info(f"stream_images_request={MessageToDict(stream_images_request, preserving_proto_field_name=True, always_print_fields_with_no_presence=True)}")
    stream_images_responses = stub.StreamImages(stream_images_request)

    mpd = PulseHeightDistribution(durations_seconds, module_ids)
    last_plot_update_time = time.time()
    for response in stream_images_responses:
        # log response metadata
        if logger:
            formatted_stream_images_response = format_stream_images_response(response)
            logger.info(formatted_stream_images_response)

        # unpack pano image
        pano_image = response.pano_image
        pano_type, header, img = unpack_pano_image(pano_image)
        ph_baseline = 750

        if pano_type == 'PULSE_HEIGHT':
            img += ph_baseline
            mpd.update(img)
            curr_time = time.time()
            if curr_time - last_plot_update_time > max(plot_update_interval, 0.5):
                mpd.plot()
                last_plot_update_time = curr_time


def run_pano_image_preview(
        stub: daq_data_pb2_grpc.DaqDataStub,
        stream_movie_data: bool,
        stream_pulse_height_data: bool,
        update_interval_seconds: float,
        module_ids: list[int],
        logger: logging.Logger,
        wait_for_ready: bool = False,
):
    """Streams PanoImages from an active observing run."""
    # Create the request message
    stream_images_request = StreamImagesRequest(
        stream_movie_data=stream_movie_data,
        stream_pulse_height_data=stream_pulse_height_data,
        update_interval_seconds=update_interval_seconds,
        module_ids=module_ids,
    )
    # Make the RPC call
    logger.info(f"stream_images_request={MessageToDict(stream_images_request, preserving_proto_field_name=True, always_print_fields_with_no_presence=True)}")
    stream_images_responses = stub.StreamImages(stream_images_request, wait_for_ready=wait_for_ready)
    previewer = PanoImagePreviewer(
        stream_movie_data, stream_pulse_height_data, update_interval_seconds, logger, col_width=4.5, row_height=2.8,
    )

    # Process responses
    for stream_images_response in stream_images_responses:
        # log response metadata
        formatted_response = format_stream_images_response(stream_images_response)
        logger.info(formatted_response)

        pano_image = stream_images_response.pano_image
        pano_type, header, img = unpack_pano_image(pano_image)
        module_id = pano_image.module_id
        previewer.update(pano_image, pano_type, header, img, module_id)


def run(args):

    init_cfg = None
    do_init = False
    if args.init_sim or args.cfg_file is not None:
        do_init = True
        if args.init_sim:
            init_cfg_path = 'config/hp_io_config_simulate_daq.json'
        elif args.cfg_file:
            init_cfg_path = f'config/{args.cfg_file}'
        else:
            init_cfg_path = None

        # try to open the config file
        if init_cfg_path is not None and not os.path.exists(init_cfg_path):
            logging.error(f"Config file not found: '{os.path.abspath(init_cfg_path)}'")
            sys.exit(1)
        else:
            with open(init_cfg_path, "r") as f:
                init_cfg = json.load(f)

    do_plot = args.plot_view or args.plot_phdist
    module_ids = args.module_ids
    if args.plot_phdist:
        if len(module_ids) == 0:
            logging.warning("no module_ids specified, using data from all modules to make ph distribution")
        elif len(module_ids) > 1:
            logging.warning("more than one module_id specified to make ph distribution")

    port = 50051
    logger = make_rich_logger(__name__, level=logging.INFO)
    connection_target = f"{args.host}:{port}"
    logger.info(f"connection_target={repr(connection_target)}")
    try:
        with grpc.insecure_channel(connection_target) as channel:
            stub = daq_data_pb2_grpc.DaqDataStub(channel)
            print("-------------- ServerReflection --------------")
            reflect_services(channel)

            if do_init:
                print("-------------- InitHpIo --------------")
                init_hp_io(
                    stub,
                    data_dir=init_cfg['data_dir'],
                    update_interval_seconds=init_cfg['update_interval_seconds'],
                    simulate_daq=init_cfg['simulate_daq'],
                    force=init_cfg['force'],
                    data_products=init_cfg['data_products'],
                    timeout=15.0,
                    logger=logger
                )

            if do_plot:
                print("-------------- StreamImages --------------")
                if args.plot_view:
                    run_pano_image_preview(
                        stub,
                        stream_movie_data=True,
                        stream_pulse_height_data=True,
                        update_interval_seconds=np.random.uniform(0.5, 1.5),
                        module_ids=module_ids,
                        wait_for_ready=True,
                        logger=logger
                    )

                elif args.plot_phdist:
                    run_pulse_height_distribution(
                        stub,
                        plot_update_interval=0.25,
                        durations_seconds= (10, 30, 60),
                        module_ids=module_ids,
                        logger=logger
                    )
                else:
                    raise ValueError("Invalid plot")
    except KeyboardInterrupt:
        logger.info(f"'^C' received, closing connection to the DaqData server at {repr(connection_target)}")
    except grpc.RpcError as rpc_error:
        logger.error(f"{type(rpc_error)}\n{repr(rpc_error)}")


def signal_handler(signum, frame):
    print(f"Signal {signum} received, exiting...")
    sys.exit(0)



if __name__ == "__main__":
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT]:
        signal.signal(sig, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        help="daq_data server hostname or IP address. Default: 'localhost'",
        # default="10.0.0.60"
        default="localhost"
    )
    parser.add_argument(
        "--init",
        help="Send an InitHpIO request to configure the hp_io thread from the file [CFG] in config/ to track an in-progress run directory",
        type=str,
        dest="cfg_file"
    )
    parser.add_argument(
        "--init-sim",
        help="Send an InitHpIo request to configure the hp_io thread to track a simulated run directory",
        action="store_true",
    )

    parser.add_argument(
        "--plot-view",
        help="Make a live data previewer",
        action="store_true",
    )

    parser.add_argument(
        "--plot-phdist",
        help="Make a live pulse-height distribution for the specified module id",
        action="store_true",
    )

    parser.add_argument(
        "--module-ids",
        help="If empty, data from all modules is returned. If non-empty, only data from the specified modules are returned",
        nargs="*",
        type=int
    )

    # run(host="10.0.0.60")
    args = parser.parse_args()
    run(args)
