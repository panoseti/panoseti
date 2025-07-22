#!/usr/bin/env python3

import signal
import argparse
import json
import logging
import os.path
from collections import deque
import time
from datetime import datetime
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
import textwrap
sys.path.append("../../util")
import pff


class PulseHeightDistribution:
    def __init__(self, durations_seconds, module_ids):
        self.durations = durations_seconds
        self.module_ids = list(module_ids)
        self.n_durations = len(self.durations)

        # For each module: list of deques per duration window
        self.start_times = {mod: [time.time() for _ in range(self.n_durations)]
                            for mod in self.module_ids}
        self.hist_data = {mod: [deque() for _ in range(self.n_durations)]
                          for mod in self.module_ids}
        self.vmins = {mod: [float('inf')] * self.n_durations for mod in self.module_ids}
        self.vmaxs = {mod: [float('-inf')] * self.n_durations for mod in self.module_ids}

        # Preallocate colors for distinct modules
        palette = sns.color_palette('husl', n_colors=len(self.module_ids))
        self.module_colors = {mod: palette[i] for i, mod in enumerate(self.module_ids)}

        # Figure/axes: one axis per duration window
        height = max(2.9 * self.n_durations, 6)
        plt.ion()
        self.fig, self.axes = plt.subplots(self.n_durations, 1, figsize=(6, height))
        if self.n_durations == 1:
            self.axes = [self.axes]

    def update(self, image, module_id):
        if module_id not in self.hist_data:
            # Dynamically add support for new modules if needed
            self.module_ids.append(module_id)
            self.start_times[module_id] = [time.time()] * self.n_durations
            self.hist_data[module_id] = [deque() for _ in range(self.n_durations)]
            self.vmins[module_id] = [float('inf')] * self.n_durations
            self.vmaxs[module_id] = [float('-inf')] * self.n_durations
            # Assign a color (expand palette if many modules)
            palette = sns.color_palette('husl', n_colors=len(self.module_ids))
            self.module_colors[module_id] = palette[len(self.module_ids)-1]

        max_pixel = int(np.max(image))
        now = time.time()
        for i, duration in enumerate(self.durations):
            if now - self.start_times[module_id][i] > duration:
                self.hist_data[module_id][i].clear()
                self.start_times[module_id][i] = now
                self.vmins[module_id][i] = float('inf')
                self.vmaxs[module_id][i] = float('-inf')
            self.hist_data[module_id][i].append(max_pixel)
            self.vmins[module_id][i] = min(self.vmins[module_id][i], max_pixel)
            self.vmaxs[module_id][i] = max(self.vmaxs[module_id][i], max_pixel)

    def plot(self):
        for i, duration in enumerate(self.durations):
            ax = self.axes[i]
            ax.clear()

            # Compute last refresh (latest start_time) for this duration window
            all_refresh = [
                self.start_times[mod][i] for mod in self.module_ids if self.hist_data[mod][i]
            ]
            if all_refresh:
                last_refresh_unix = max(all_refresh)
                last_refresh = datetime.fromtimestamp(last_refresh_unix).strftime('%Y-%m-%d %H:%M:%S')
            else:
                last_refresh = "Never"

            # Axis limits
            mins = [self.vmins[mod][i] for mod in self.module_ids if self.hist_data[mod][i]]
            maxs = [self.vmaxs[mod][i] for mod in self.module_ids if self.hist_data[mod][i]]
            vmin = min(mins) if mins else 0
            vmax = max(maxs) if maxs else 1

            for mod in self.module_ids:
                values = self.hist_data[mod][i]
                if values:
                    sns.histplot(
                        list(values),
                        bins=100,
                        kde=False,
                        stat='density',
                        element='step',
                        label=f'Module {mod}',
                        color=self.module_colors[mod],
                        ax=ax,
                    )
            ax.set_xlim(vmin - 10, vmax + 10)
            ax.set_title(
                f"Refresh interval = {duration}s | Last refresh = {last_refresh}",
                fontsize=12,
            )
            ax.set_xlabel("ADC Value")
            ax.set_ylabel("Density")
            ax.legend(title="Module", fontsize=9, title_fontsize=10)
        self.fig.suptitle( f"Distribution of Max Pulse-Heights")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class PanoImagePreviewer:
    def __init__(
            self,
            stream_movie_data: bool,
            stream_pulse_height_data: bool,
            update_interval_seconds: float,
            module_id_whitelist: list[int],
            logger: logging.Logger,
            text_width=25,
            font_size=7,
            row_height=2.8,
            window_size=100,
    ) -> None:
        self.stream_movie_data = stream_movie_data
        self.stream_pulse_height_data = stream_pulse_height_data
        self.update_interval_seconds = update_interval_seconds
        self.module_id_whitelist = module_id_whitelist
        self.logger = logger

        self.seen_modules = set()
        self.axes_map = {}
        self.cbar_map = {}
        self.im_map = {}
        self.window_size = window_size
        self.max_pix_map = {'PULSE_HEIGHT': deque(maxlen=self.window_size), 'MOVIE': deque(maxlen=self.window_size)}
        self.min_pix_map = {'PULSE_HEIGHT': deque(maxlen=self.window_size), 'MOVIE': deque(maxlen=self.window_size)}

        self.fig = None
        self.text_width = text_width
        self.font_size = font_size
        # self.cmap = np.random.choice(['magma', 'viridis', 'rocket', 'mako'])
        self.cmap = 'plasma'
        self.row_height = row_height
        self.num_rescale = 0

    def setup_layout(self, modules):
        """Sets up subplot layout: one row per module, two columns (PH left, Movie right)."""
        if self.fig is not None:
            plt.close(self.fig)
        modules = sorted(modules)
        n_modules = len(modules)
        self.fig, axs = plt.subplots(n_modules, 2, figsize=(self.row_height * 2.2, self.row_height * n_modules))
        if n_modules == 1:
            axs = np.array([axs])  # one row per module

        self.num_rescale = 0
        self.axes_map.clear()
        self.cbar_map.clear()
        self.im_map.clear()
        for row, module_id in enumerate(modules):
            self.axes_map[(module_id, 'PULSE_HEIGHT')] = axs[row, 0]
            self.axes_map[(module_id, 'MOVIE')] = axs[row, 1]

            im_ph = axs[row, 0].imshow(np.zeros((32, 32)), cmap=self.cmap)
            self.im_map[(module_id, 'PULSE_HEIGHT')] = im_ph
            im_mov = axs[row, 1].imshow(np.zeros((32, 32)), cmap=self.cmap)
            self.im_map[(module_id, 'MOVIE')] = im_mov

            # Create a divider for each axis for inline colorbar
            divider_ph = make_axes_locatable(axs[row, 0])
            cax_ph = divider_ph.append_axes('right', size='5%', pad=0.05)
            cbar_ph = self.fig.colorbar(im_ph, cax=cax_ph)
            self.cbar_map[(module_id, 'PULSE_HEIGHT')] = cbar_ph

            divider_mov = make_axes_locatable(axs[row, 1])
            cax_mov = divider_mov.append_axes('right', size='5%', pad=0.05)
            cbar_mov = self.fig.colorbar(im_mov, cax=cax_mov)
            self.cbar_map[(module_id, 'MOVIE')] = cbar_mov
        self.fig.tight_layout()
        plt.ion()
        plt.show()

    def update(self, frame_number, file, pano_type, header, img, module_id):
        # check if this module is new
        if module_id not in self.seen_modules:
            self.seen_modules.add(module_id)
            self.setup_layout(self.seen_modules)

        # update dynamic min and max data dequeues
        self.max_pix_map[pano_type].append(np.max(img))
        self.min_pix_map[pano_type].append(np.min(img))
        vmax = np.quantile(self.max_pix_map[pano_type], 0.95)
        vmin = np.quantile(self.min_pix_map[pano_type], 0.05)
        im = self.im_map[(module_id, pano_type)]
        im.set_data(img)
        im.set_clim(vmin, vmax)

        cbar = self.cbar_map.get((module_id, pano_type))
        cbar.ax.tick_params(labelsize=8)
        cbar.locator = MaxNLocator(nbins=6)
        cbar.update_ticks()
        ax = self.axes_map.get((module_id, pano_type))
        if ax is None:
            return

        # Prepare axis title with details
        ax_title = (f"{pano_type}"
                    + ("\n" if 'quabo_num' not in header else f": Q{int(header['quabo_num'])}\n")
                    + f"unix_t = {header['pandas_unix_timestamp'].time()}\n"
                    + f"frame_no = {frame_number}\n")
        ax_title += textwrap.fill(f"file = {file}", width=self.text_width)

        ax.set_title(ax_title, fontsize=self.font_size)
        ax.tick_params(axis='both', which='major', labelsize=8, length=4, width=1)

        start = pff.parse_name(file)['start']
        if len(self.module_id_whitelist) > 0:
            plt_title = f"Obs data from {start}, module_ids={set(self.module_id_whitelist)} [filtered]"
        else:
            plt_title = f"Obs data from {start}, module_ids={self.seen_modules} [all]"
        if self.num_rescale < len(self.seen_modules) * 3:
            self.fig.tight_layout()
            self.num_rescale += 1
        self.fig.suptitle(plt_title)
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

    ph_dist = PulseHeightDistribution(durations_seconds, module_ids)
    last_plot_update_time = time.time()
    for response in stream_images_responses:
        # log response metadata
        if logger:
            formatted_stream_images_response = format_stream_images_response(response)
            logger.info(formatted_stream_images_response)

        # unpack pano image
        pano_image = response.pano_image
        module_id, pano_type, header, img = unpack_pano_image(pano_image)

        if pano_type == 'PULSE_HEIGHT':
            # img += np.random.poisson(lam=50, size=img.shape)
            ph_dist.update(img, module_id)
            curr_time = time.time()
            if curr_time - last_plot_update_time > plot_update_interval:
                ph_dist.plot()
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
        stream_movie_data, stream_pulse_height_data, update_interval_seconds,
        module_ids, logger, row_height=3, font_size=6, text_width=30, window_size=1000
    )

    # Process responses
    for stream_images_response in stream_images_responses:
        # log response metadata
        formatted_response = format_stream_images_response(stream_images_response)
        logger.info(formatted_response)

        pano_image = stream_images_response.pano_image
        module_id, pano_type, header, img = unpack_pano_image(stream_images_response.pano_image)
        previewer.update(pano_image.frame_number, pano_image.file, pano_type, header, img, module_id)


def run(args):
    logger = make_rich_logger(__name__, level=logging.INFO)
    hp_io_cfg = None
    do_init_hp_io = False
    if args.init_sim or args.cfg_path is not None:
        do_init_hp_io = True
        if args.init_sim:
            hp_io_cfg_path = 'config/hp_io_config_simulate_daq.json'
        elif args.cfg_file:
            hp_io_cfg_path = f'{args.cfg_file}'
        else:
            hp_io_cfg_path = None

        # try to open the config file
        if hp_io_cfg_path is not None and not os.path.exists(hp_io_cfg_path):
            logging.error(f"Config file not found: '{os.path.abspath(hp_io_cfg_path)}'")
            sys.exit(1)
        else:
            with open(hp_io_cfg_path, "r") as f:
                hp_io_cfg = json.load(f)

    do_plot = args.plot_view or args.plot_phdist
    module_ids = args.module_ids
    if args.plot_phdist:
        if len(module_ids) == 0:
            logging.warning("no module_ids specified, using data from all modules to make ph distribution")
        elif len(module_ids) > 1:
            logging.warning("more than one module_id specified to make ph distribution")

    port = 50051
    connection_target = f"{args.host}:{port}"
    try:
        with grpc.insecure_channel(connection_target) as channel:
            stub = daq_data_pb2_grpc.DaqDataStub(channel)
            print("-------------- ServerReflection --------------")
            logger.info(f"connection_target={repr(connection_target)}")
            reflect_services(channel)

            if do_init_hp_io:
                print("-------------- InitHpIo --------------")
                module_id_whitelist = []
                if 'module_ids' in hp_io_cfg:
                    module_id_whitelist = hp_io_cfg['module_ids']
                init_hp_io(
                    stub,
                    data_dir=hp_io_cfg['data_dir'],
                    update_interval_seconds=hp_io_cfg['update_interval_seconds'],
                    simulate_daq=hp_io_cfg['simulate_daq'],
                    force=hp_io_cfg['force'],
                    module_ids=module_id_whitelist,
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
                        update_interval_seconds=2.0, #np.random.uniform(1.0, 1.0),
                        module_ids=module_ids,
                        wait_for_ready=True,
                        logger=logger
                    )

                elif args.plot_phdist:
                    run_pulse_height_distribution(
                        stub,
                        plot_update_interval=1.0,
                        durations_seconds= (10, 60, 600),
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
        help="DaqData server hostname or IP address. Default: 'localhost'",
        # default="10.0.0.60"
        default="localhost"
    )
    parser.add_argument(
        "--init",
        help="initialize the hp_io thread from the file [CFG] in config/ to track an in-progress run directory",
        type=str,
        dest="cfg_path"
    )
    parser.add_argument(
        "--init-sim",
        help="initialize the hp_io thread to track a simulated run directory",
        action="store_true",
    )

    parser.add_argument(
        "--plot-view",
        help="whether to create a live data previewer",
        action="store_true",
    )

    parser.add_argument(
        "--plot-phdist",
        help="whether to create a live pulse-height distribution for the specified module id",
        action="store_true",
    )

    parser.add_argument(
        "--module-ids",
        help="whitelist for the module ids to stream data from. If empty, data from all available modules are returned.",
        nargs="*",
        type=int,
        default=[],
    )

    # run(host="10.0.0.60")
    args = parser.parse_args()
    run(args)
