#! /usr/bin/env python3


import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import seaborn as sns

from cloud_utils import get_pd_data, get_file_info_array, get_img_spike_dir

sys.path.append("../util")
import config_file
import pff
import image_quantiles
import show_pff


DATA_DIR = '/Users/nico/Downloads/panoseti_test_data/obs_data'
RUN_DIR = 'obs_Lick.start_2023-07-19T06:07:34Z.runtype_sci-obs.pffd'

data_dir = DATA_DIR + '/data'
run_dir = RUN_DIR

data_config = config_file.get_data_config(f'{data_dir}/{run_dir}')
integration_time = float(data_config["image"]["integration_time_usec"]) * 10 ** (-6)  # 100 * 10**(-6)
step_size = 1
sigma = 10
module = 1
file_info_array = get_file_info_array(data_dir, run_dir, module)
start_unix_t = file_info_array[0]['first_unix_t']

analysis_info = {
    "run_dir": run_dir,
    "sigma": sigma,
    "step_size": step_size
}


def show_spikes(data_dir, run_dir, file_info_array, integration_time, data_fpath):
    pd_data = get_pd_data(file_info_array, data_fpath)
    start_unix_t = file_info_array[0]['first_unix_t']
    print(start_unix_t)
    times = pd_data.loc[:, 'Elapsed Run Time (sec)'] + start_unix_t
    print(times)
    frame_time = integration_time * 10 ** (-6)
    image_size = 32
    quantile = 0.1
    # Find file containing desired frame
    for i in range(len(times)):
        t = times[i]
        print(t)
        for finfo in file_info_array:
            if not (finfo['first_unix_t'] <= t <= finfo['last_unix_t']):
                continue
            fname = f'{data_dir}/{run_dir}/{finfo["fname"]}'
            bytes_per_pixel = finfo["bytes_per_pixel"]
            bytes_per_image = bytes_per_pixel * image_size**2
            [min, max] = image_quantiles.get_quantiles(
                fname, image_size, bytes_per_pixel, quantile
            )
            print('pixel 10/90 percentiles: %d, %d' % (min, max))
            with open(fname, 'rb') as f:
                #print(f.tell())
                pff.time_seek(f, frame_time, bytes_per_image, t)
                #print(f.tell())
                j = pff.read_json(f)
                if not j:
                    print('reached EOF')
                    break
                show_pff.print_json(j.encode(), False, verbose=False)
                img = pff.read_image(f, image_size, bytes_per_pixel)
                show_pff.image_as_text(img, image_size, bytes_per_pixel, min, max)
                print(pd_data.iloc[i].to_string())
                x = input("Enter for next frame, 'q' to quit: ")
                if x == 'q':
                    return

img_spike_dir = get_img_spike_dir(file_info_array[0], file_info_array[-1], analysis_info)

show_spikes(data_dir, run_dir, file_info_array, integration_time, f'image_spikes/{img_spike_dir}/spike_data.csv')