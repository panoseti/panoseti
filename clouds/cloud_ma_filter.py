#! /usr/bin/env python3
"""
Moving average filter
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import math

sys.path.append("../util")
import config_file
import pff

DATA_DIR = '/Users/nico/Downloads/test_data/obs_data'
RUN_DIR = 'obs_Lick.start_2023-07-19T06:07:34Z.runtype_sci-obs.pffd'
fname = 'start_2023-07-19T06_07_59Z.dp_img16.bpp_2.module_1.seqno_0.pff'

data_dir = DATA_DIR + '/data'
run_dir = RUN_DIR

def get_next_frame(f, frame_size, bytes_per_pixel, step_size):
    """Returns the next image frame and json header from f."""
    j = json.loads(pff.read_json(f))
    img = pff.read_image(f, 32, bytes_per_pixel)
    f.seek((step_size - 1) * frame_size, os.SEEK_CUR) # Skip step_size - 1 images
    return img, j


def get_img_brightness_signal(f, data, arr_offset, img_size, bytes_per_pixel, step_size):
    frame_size, nframes, first_unix_t, last_unix_t = pff.img_info(f, img_size)
    f.seek(0)
    for i in range(nframes // step_size):
        img, j = get_next_frame(f, frame_size, bytes_per_pixel, step_size)
        data[arr_offset + i] = np.sum(img)


def process_file(fname, data, arr_offset, step_size):
    with open(f'{data_dir}/{run_dir}/{fname}', 'rb') as f:
        file_attrs = pff.parse_name(fname)
        bytes_per_pixel = int(file_attrs['bpp'])
        img_size = bytes_per_pixel * 1024
        get_img_brightness_signal(f, data, arr_offset, img_size, bytes_per_pixel, step_size)


def get_file_nframes(files_to_process):
    file_nframes = []
    for fname in files_to_process:
        with open(f'{data_dir}/{run_dir}/{fname}', 'rb') as f:
            file_attrs = pff.parse_name(fname)
            bytes_per_pixel = int(file_attrs['bpp'])
            img_size = bytes_per_pixel * 1024
            frame_size, nframes, first_unix_t, last_unix_t = pff.img_info(f, img_size)
        file_nframes += [nframes]
    return file_nframes


def get_empty_data_array(file_nframes, step_size):
    data_size = 0
    for i in range(len(file_nframes)):
        data_size += file_nframes[i] // step_size
    return np.zeros(data_size)


data_config = config_file.get_data_config(f'{data_dir}/{run_dir}')
integration_time = float(data_config["image"]["integration_time_usec"]) * 10 ** (-6)  # 100 * 10**(-6)
step_size = 4096

# Assumes only one module in directory (for now)
files_to_process = []
for fname in os.listdir(f'{data_dir}/{run_dir}'):
    if pff.is_pff_file(fname) and pff.pff_file_type(fname) in ('img16', 'img8'):
        files_to_process.append(fname)


def fname_sort_key(fname):
    file_attrs = pff.parse_name(fname)
    return int(file_attrs["seqno"])


files_to_process.sort(key=fname_sort_key)
# files_to_process = files_to_process[:5]
file_nframes = get_file_nframes(files_to_process)
data = get_empty_data_array(file_nframes, step_size)

arr_offset = 0
for i in range(len(files_to_process)):
    print(f"Processing {files_to_process[i]}")
    process_file(files_to_process[i], data, arr_offset, step_size)
    arr_offset += file_nframes[i] // step_size


# Moving average impulse responses
MA5 = np.ones(5) / 5
MA25 = np.ones(25) / 25
MA75 = np.ones(75) / 75
MA150 = np.ones(150) / 150

# Moving averages
y5 = np.convolve(data, MA5, "same")
y25 = np.convolve(data, MA25, "same")
y75 = np.convolve(data, MA75, "same")
y150 = np.convolve(data, MA150, "same")

x = np.arange(len(data)) * step_size * integration_time

plt.plot(x, data)
plt.plot(x, y5)
plt.plot(x, y25)
plt.plot(x, y75)
plt.plot(x, y150)


def ema_filter(length):
    alpha = 2/(length+1)
    # Create and return an EMA filter with length "length"
    h = (1 - alpha) ** np.arange(length)
    return h / np.sum(h)

print(len(data))

# plt.xlim([75 * step_size * integration_time, (len(x) - 75) * step_size * integration_time])
plt.xlim([0, len(x) * step_size * integration_time])
ylow = np.mean(data) - 5 * np.std(data)
yhigh = np.mean(data) + 5 * np.std(data)
plt.ylim([ylow, yhigh])
# plt.ylim([min(data) * 0.999, max(data) * 1.001])
plt.legend(('Total Brightness', '5-Point Average', '25-Point Average', '75-Point Average', '150-Point Average'))
# plt.legend(('Total Brightness', '25-Point Average', '75-Point Average'))

plt.ylabel("Total counts")
plt.xlabel("Seconds since start of run")
# plt.xlabel("Frame index")
plt.title(f"Moving Averaged Movie Frame Brightness @ 100 µs (frame step size={step_size})")
plt.show()


