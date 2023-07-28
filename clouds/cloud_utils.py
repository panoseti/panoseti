"""
Utility functions used by the cloud detection pipeline.
"""
import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append("../util")
import config_file
import pff


def get_next_frame(f, frame_size, bytes_per_pixel, step_size):
    """Returns the next image frame and json header from f."""
    j = json.loads(pff.read_json(f))
    img = pff.read_image(f, 32, bytes_per_pixel)
    f.seek((step_size - 1) * frame_size, os.SEEK_CUR)   # Skip (step_size - 1) images
    return img, j


def get_file_info_array(data_dir, run_dir, files_to_process):
    """Returns an array of dictionaries storing the attributes of all the files to process."""
    file_attrs = [None] * len(files_to_process)
    for i in range(len(files_to_process)):
        fname = files_to_process[i]
        attrs = {"fname": fname}
        with open(f'{data_dir}/{run_dir}/{fname}', 'rb') as f:
            parsed_name = pff.parse_name(fname)
            bytes_per_pixel = int(parsed_name['bpp'])
            img_size = bytes_per_pixel * 1024
            attrs["frame_size"], attrs["nframes"], attrs["first_unix_t"], attrs["last_unix_t"] = pff.img_info(f, img_size)
            attrs["img_size"] = img_size
            attrs["bytes_per_pixel"] = bytes_per_pixel
            attrs["module"] = int(parsed_name["module"])
            attrs["seqno"] = int(parsed_name['seqno'])
        file_attrs[i] = attrs
    return file_attrs

def get_PDT_timestamp(unix_t):
    dt = datetime.datetime.fromtimestamp(unix_t, datetime.timezone(datetime.timedelta(hours=-7)))
    return dt.strftime("%m/%d/%Y, %H:%M:%S")

