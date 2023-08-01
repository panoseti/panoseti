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

from cloud_utils import get_file_info_array, get_next_frame, get_PDT_timestamp

DATA_DIR = '/Users/nico/Downloads/test_data/obs_data'
RUN_DIR = 'obs_Lick.start_2023-07-19T06:07:34Z.runtype_sci-obs.pffd'
fname = 'start_2023-07-19T06_07_59Z.dp_img16.bpp_2.module_1.seqno_0.pff'

data_dir = DATA_DIR + '/data'
run_dir = RUN_DIR


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


def fname_sort_key(fname):
    parsed_name = pff.parse_name(fname)
    return int(parsed_name["seqno"])


def ema_filter(length):
    alpha = 2/(length+1)
    # Create and return an EMA filter with length "length"
    h = (1 - alpha) ** np.arange(length)
    return h / np.sum(h)


def get_img_spike_dir(file_info0, file_info1, analysis_info):
    return f"{analysis_info['run_dir']}" \
               f".module_{file_info0['module']}" \
               f".seqno0_{file_info0['seqno']}" \
               f".seqno1_{file_info1['seqno']}" \
               f".window-width_{analysis_info['window_width']}" \
               f".step-size_{analysis_info['step_size']}"


def gen_npy_fname(img_spike_dir):
    npy_fname = f"image_spikes/{img_spike_dir}/data.npy"
    path = Path(f"image_spikes/{img_spike_dir}/{fname}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return npy_fname
    # return f"image_spikes/{get_img_spike_dir(run_dir, module, seqno0, seqno1, sigma, window_width, step_size)}.npy"


def get_data(img_spike_dir, step_size):
    # Save reduced data to file
    npy_fname = gen_npy_fname(img_spike_dir)
    if os.path.exists(npy_fname):
        data_arr = np.load(npy_fname)
        return data_arr
    itr_info = {
        "data_offset": 0,
        "fstart_offset": 0  # Ensures frame step size across files
    }
    data_arr = get_empty_data_array(file_info_array, step_size)
    for i in range(len(files_to_process)):
        print(f"Processing {files_to_process[i]}")
        file_info = file_info_array[i]
        process_file(file_info, data_arr, itr_info, step_size)
        itr_info['data_offset'] += file_info["nframes"] // step_size

    np.save(npy_fname, data_arr)
    return data_arr


data_config = config_file.get_data_config(f'{data_dir}/{run_dir}')
integration_time = float(data_config["image"]["integration_time_usec"]) * 10 ** (-6)  # 100 * 10**(-6)
step_size = 512
window_width = 10**4
sigma = 4

analysis_info = {
    "run_dir": run_dir,
    "sigma": sigma,
    "window_width": window_width,
    "step_size": step_size
}

# Assumes only one module in directory (for now)
files_to_process = []
for fname in os.listdir(f'{data_dir}/{run_dir}'):
    if pff.is_pff_file(fname) and pff.pff_file_type(fname) in ('img16', 'img8'):
        files_to_process.append(fname)

files_to_process.sort(key=fname_sort_key)  # Sort files_to_process in ascending order by file sequence number
# files_to_process = files_to_process[:len(files_to_process) - 18]
files_to_process = files_to_process[:len(files_to_process) - 18]
file_info_array = get_file_info_array(data_dir, run_dir, files_to_process)

start_unix_t = file_info_array[0]['first_unix_t']


img_spike_dir = get_img_spike_dir(file_info_array[0], file_info_array[-1], analysis_info)
data = get_data(img_spike_dir, step_size)
x = np.arange(len(data)) * step_size * integration_time

print(img_spike_dir)

spike_centers = []
spikes = pd.DataFrame(
    columns=["Timestamp", "Elapsed Run Time (sec)", "Total Counts", "Local Mean", "Local Std", "Local Z-score"]
)

MAfilter = np.ones(500) / 500
y5 = np.convolve(data, MAfilter, "same")

for i in range(window_width, len(data) - window_width):
    local_mean = np.mean(data[i - window_width:i + window_width])
    local_std = np.std(data[i - window_width:i + window_width])
    zscore = (data[i] - local_mean) / local_std
    if abs(zscore) > sigma:
        obs_time = get_PDT_timestamp(start_unix_t + x[i])
        spikes.loc[len(spikes.index)] = [obs_time, x[i], data[i], local_mean, local_std, zscore]
        spike_centers.append(i)

title = f"Movie Frames with Total Counts >{sigma} Sigma Relative to Local {2 * window_width}-Point Sample (integration time={round(integration_time * 10 ** 6)} µs, step_size={step_size})"
sns.set_style("darkgrid")
sns.color_palette("flare_r", as_cmap=True)
# ax = sns.scatterplot(data=spikes, x="Elapsed Run Time (sec)", y="Local Z-score", hue="Local Z-score", palette="flare_r")
ax = sns.scatterplot(data=spikes, x="Elapsed Run Time (sec)", y="Total Counts", hue="Total Counts", palette="flare_r")
plt.title(label=title[:len(title)//2] + '\n' + title[len(title)//2:])
# # Put a legend below current axis
#plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')#, label='Total Frame Counts')
#plt.legend(markerscale=1)
plt.tight_layout()
plt.savefig(f"image_spikes/{img_spike_dir}/seaborn_plot.svg")#, bbox_inches="tight")
# .move_legend(p, "upper left", bbox_to_anchor=(1, 1))

mean = np.mean(data)
std = np.std(data)

str_out = f"{data_dir}/{run_dir}:\n" \
          f"\t{files_to_process[0]}\n" \
          f"\t{files_to_process[-1]}\n\n" \
          f"{title}\n\n" \
          f"{spikes.to_string()}\n\n" \
          f"# frames processed = {len(data)}\n" \
          f"mean={mean}, std={std}"

filepath = Path(f"image_spikes/{img_spike_dir}/spike_data.csv")
filepath.parent.mkdir(parents=True, exist_ok=True)
spikes.to_csv(filepath)

with open(f"image_spikes/{img_spike_dir}/summary.txt", "w") as f:
    f.write(str_out)



print(str_out)




#p.show()

# for i in spike_centers:
#     sns.histplot(data=data[i - width:i + width], stat="density", )
#     plt.show()












# ---- Plot MA and EMAs ----

# Moving average impulse responses
MA5 = np.ones(5) / 5
MA25 = np.ones(25) / 25
MA75 = np.ones(75) / 75
MA500 = np.ones(500) / 500
MA1000 = np.ones(1000) / 1000
MA5000 = np.ones(5000) / 5000
MAVar_size = len(data) // 20
MAVar = np.ones(MAVar_size) / MAVar_size

EMA_size = len(data) // 20
EMA = ema_filter(EMA_size)

# Moving averages
y5 = np.convolve(data, MA5, "same")
y25 = np.convolve(data, MA25, "same")
y75 = np.convolve(data, MA75, "same")
y500 = np.convolve(data, MA500, "same")
y1000 = np.convolve(data, MA1000, "same")
y5000 = np.convolve(data, MA5000, "same")
yVar = np.convolve(data, MAVar, "same")
yEMA = np.convolve(data, EMA, "same")

plt.plot(x, data)
#plt.plot(x, y5)
#plt.plot(x, y25)
#plt.plot(x, y1000)
#plt.plot(x, yVar)
#plt.plot(x, yEMA)



# plt.xlim([75 * step_size * integration_time, (len(x) - 75) * step_size * integration_time])
plt.xlim([0, len(x) * step_size * integration_time])
ylow = np.mean(data) - 5 * np.std(data)
yhigh = np.mean(data) + 5 * np.std(data)
#plt.ylim([ylow, yhigh])
# plt.ylim([min(data) * 0.999, max(data) * 1.001])
#plt.legend(('Total Brightness', '5-Point Average', '25-Point Average', '75-Point Average', f'{MAVar_size}-Point Average', f'{EMA_size}-Point EMA'))
#plt.legend(('Total Brightness', '10000-Point Average', f'{MAVar_size}-Point Average', f'{EMA_size}-Point EMA'))

# plt.ylabel("Total counts")
#plt.xlabel("Seconds since start of run")
# plt.xlabel("Frame index")
#plt.title(f"Moving Averaged Movie Frame Brightness @ 100 µs (frame step size={step_size})")

plt.show()


