#! /usr/bin/env python3
"""
Moving average filter
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import seaborn as sns
import seaborn.objects as so
import pandas as pd
import math
import datetime
from pathlib import Path

sys.path.append("../util")
import config_file
import pff

from cloud_utils import get_file_info_array, get_next_frame, get_PDT_timestamp, get_img_spike_dir

DATA_DIR = '/Users/nico/Downloads/panoseti_test_data/obs_data'
RUN_DIR = 'obs_Lick.start_2023-07-19T06:07:34Z.runtype_sci-obs.pffd'

data_dir = DATA_DIR + '/data'
run_dir = RUN_DIR

data_config = config_file.get_data_config(f'{data_dir}/{run_dir}')
integration_time = float(data_config["image"]["integration_time_usec"]) * 10 ** (-6)  # 100 * 10**(-6)
step_size = 1
sigma = 10
module = 1

analysis_info = {
    "run_dir": run_dir,
    "sigma": sigma,
    "step_size": step_size
}


def process_file(file_info, data, itr_info, step_size):
    """On a sample of the frames in the file represented by file_info, add the total
    image brightness to the data array beginning at data_offset."""
    with open(f"{data_dir}/{run_dir}/{file_info['fname']}", "rb") as f:
        # Start file pointer with an offset based on the previous file -> ensures even frame sampling
        f.seek(
            itr_info['fstart_offset'] * file_info['frame_size'],
            os.SEEK_CUR
        )
        new_nframes = file_info['nframes'] - itr_info['fstart_offset']
        for i in range(new_nframes // step_size):
            img, j = get_next_frame(f,
                                    file_info['frame_size'],
                                    file_info['bytes_per_pixel'],
                                    step_size)
            data[itr_info['data_offset'] + i] = np.sum(img)
        itr_info['fstart_offset'] = file_info['nframes'] - (new_nframes // step_size) * step_size


def get_empty_data_array(file_attrs, step_size):
    data_size = 0
    for i in range(len(file_attrs)):
        data_size += file_attrs[i]['nframes'] // step_size
    return np.zeros(data_size)


def ema_filter(length):
    alpha = 2/(length+1)
    # Create and return an EMA filter with length "length"
    h = (1 - alpha) ** np.arange(length)
    return h / np.sum(h)


def gen_npy_fname(analysis_dir):
    npy_fname = f"{analysis_dir}/data.npy"
    return npy_fname


def get_data(file_info_array, analysis_dir, step_size):
    # Save reduced data to file
    print(img_spike_dir)
    npy_fname = gen_npy_fname(analysis_dir)
    if os.path.exists(npy_fname):
        data_arr = np.load(npy_fname)
        return data_arr
    itr_info = {
        "data_offset": 0,
        "fstart_offset": 0  # Ensures frame step size across files
    }
    data_arr = get_empty_data_array(file_info_array, step_size)
    for i in range(len(file_info_array)):
        print(f"Processing {file_info_array[i]['fname']}")
        file_info = file_info_array[i]
        process_file(file_info, data_arr, itr_info, step_size)
        itr_info['data_offset'] += file_info["nframes"] // step_size

    np.save(npy_fname, data_arr)
    return data_arr


def get_files_to_process(data_dir, run_dir, module):
    files_to_process = []
    for fname in os.listdir(f'{data_dir}/{run_dir}'):
        if pff.is_pff_file(fname) and pff.pff_file_type(fname) in ('img16', 'img8'):
            files_to_process.append(fname)
    return files_to_process

# Process file info
file_info_array = get_file_info_array(data_dir, run_dir, module)
start_unix_t = file_info_array[0]['first_unix_t']

# Load or compute reduced image data
img_spike_dir = get_img_spike_dir(file_info_array[0], file_info_array[-1], analysis_info)
analysis_dir = f"image_spikes/{img_spike_dir}"
os.makedirs(analysis_dir, exist_ok=True)
data = get_data(file_info_array, analysis_dir, step_size)
x = np.arange(len(data)) * step_size * integration_time


mean = np.mean(data)
std = np.std(data)
title = f"Module {module} Movie Frames with Total Count Z-score >{sigma}\n" \
        f"run_start={get_PDT_timestamp(start_unix_t)}\n" \
        f"integration time={round(integration_time * 10 ** 6)} µs, frame step size={step_size}, frames={len(data)}\n" \
        f"mean={round(float(mean), 3)}, std={round(float(std), 3)}"
csv_filepath = f"{analysis_dir}/spike_data.csv"
summary_filepath = f"{analysis_dir}/summary.txt"

if os.path.exists(csv_filepath) and os.path.exists(summary_filepath):
    spikes = pd.read_csv(csv_filepath)
    with open(summary_filepath, "r") as f:
        str_out = f.read()
else:
    # Compute spike locations
    spike_centers = []
    spikes = pd.DataFrame(
        columns=["Timestamp", "Elapsed Run Time (sec)", "Total Counts", "Z-score"]
    )
    for i in range(0, len(data)):
        zscore = (data[i] - mean) / std
        if abs(zscore) > sigma:
            obs_time = get_PDT_timestamp(start_unix_t + x[i])
            spikes.loc[len(spikes.index)] = [obs_time, x[i], data[i], zscore]
            spike_centers.append(i)
    spikes.to_csv(csv_filepath)

    # Generate summary table
    summary_txt = f"\n{data_dir}/{run_dir}:\n" \
              f"\tfirst: {file_info_array[0]['fname']}\n" \
              f"\tlast: {file_info_array[-1]['fname']}\n\n" \
              f"# frames processed = {len(data)}\n" \
              f"mean={mean}, std={std}\n\n" \
              f"{title}\n\n" \
              f"{spikes.to_string()}\n"
    with open(summary_filepath, "w") as f:
        f.write(summary_txt)
    print(summary_txt)

if len(spikes.index) > 0:
    sns.set_style("darkgrid")
    sns.color_palette("flare_r", as_cmap=True)
    ax = sns.scatterplot(data=spikes, x="Elapsed Run Time (sec)", y="Total Counts", hue="Total Counts", palette="flare_r")
    plt.title(label=title)
    plt.tight_layout()

l_original, = plt.plot(x, data)
#plt.xlim([0, len(x) * step_size * integration_time])
ylow = np.min(data) - 5 * std
yhigh = np.max(data) + 15 * std#100 * np.std(data)
plt.ylim(ylow, yhigh)
plt.savefig(f"{analysis_dir}/seaborn_plot.svg")#, bbox_inches="tight")
plt.show()


