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
    f.seek((step_size - 1) * frame_size, os.SEEK_CUR)
    return img, j


def get_img_brightness_signal(f, img_size, bytes_per_pixel, step_size=1):
    frame_size, nframes, first_unix_t, last_unix_t = pff.img_info(f, img_size)
    f.seek(0)

    y_len = nframes // step_size

    y = np.zeros(y_len)

    for i in range(len(y)):
        img, j = get_next_frame(f, frame_size, bytes_per_pixel, step_size)
        y[i] = np.sum(img)
    return y


with open(f'{data_dir}/{run_dir}/{fname}', 'rb') as f:
    step_size = 256
    integration_time = 100 * 10**(-6)

    file_attrs = pff.parse_name(fname)
    bytes_per_pixel = int(file_attrs['bpp'])
    img_size = bytes_per_pixel * 1024

    data = get_img_brightness_signal(f, img_size, bytes_per_pixel, step_size)

    # Moving average impulse responses
    MA5 = np.ones(5) / 5
    MA25 = np.ones(25) / 25
    MA75 = np.ones(75) / 75

    # Moving averages
    y5 = np.convolve(data, MA5, "same")
    y25 = np.convolve(data, MA25, "same")
    y75 = np.convolve(data, MA75, "same")

    x = np.arange(len(data)) * step_size * integration_time

    plt.plot(x, data)
    plt.plot(x, y5)
    plt.plot(x, y25)
    plt.plot(x, y75)

    # plt.plot(
    #     np.arange(min(1000, len(data))) * step_size * integration_time,
    #     data
    # )
    print(len(data))

    plt.xlim([100 * step_size * integration_time, (len(x) - 100) * step_size * integration_time])
    plt.ylim([min(data) * 0.999, max(data) * 1.001])
    # plt.ylim([75_000, 80_000])
    plt.legend(('Total Brightness', '5-Point Average', '25-Point Average', '75-Point Average'))

    plt.ylabel("Total counts")
    plt.xlabel("Seconds since start of run")
    # plt.xlabel("Frame index")
    plt.title(f"Moving Averaged Movie Frame Brightness @ 100 µs (frame step size={step_size})")
    plt.show()


