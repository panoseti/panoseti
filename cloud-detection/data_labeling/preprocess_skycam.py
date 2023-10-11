#! /usr/bin/env python3
# Crops the panoseti field of view in Lick all-sky camera images
# Assumes Lick webcam
#   - SC images are 480x640 pixels
#   - SC2 images are 521x765 pixels

import json

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import cv2
import os

from PIL import Image


def get_img_dirs(skycam_dir):
    """Return """
    original_skycam_dir = f'{skycam_dir}/original'
    cropped_out_dir = f'{skycam_dir}/cropped'
    pfov_out_dir = f'{skycam_dir}/pfov'
    return original_skycam_dir, cropped_out_dir, pfov_out_dir


def init_dirs(skycam_dir):
    """Initialize pre-processing directories."""
    cropped_out_dir, pfov_out_dir = get_img_dirs(skycam_dir)[1:]
    for dir_name in [skycam_dir, cropped_out_dir, pfov_out_dir]:
        os.makedirs(dir_name, exist_ok=True)


def get_corners():
    """ Read the most recent pixel corner data created by panofovlickwebc.m"""
    # TODO: search for the last astrometry calibration in the file.
    with open(pixel_data_file, 'r') as fp:
        pixel_data = json.load(fp)
        x_corners = np.round(pixel_data['SC2']['astrometry_entries'][0]['x_corners'])
        y_corners = np.round(pixel_data['SC2']['astrometry_entries'][0]['y_corners'])
        # Stack coords into an array of 2D points
        corners_4x2 = np.vstack((y_corners, x_corners)).astype(np.int64).T
        corners_4x1x2 = np.expand_dims(corners_4x2, axis=1)     # Shape for CV2 coord arrays
        return corners_4x1x2


def crop_img(img, corners, cropped_fpath):
    """Extract the panoseti FoV from a Lick all-sky camera image."""
    # Transformation code from https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
    # Find minimum bounding rectangle
    rect = cv2.minAreaRect(corners)

    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")

    # coordinate of the points in box points after the rectangle has been straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    dst_pts = np.roll(dst_pts, shift=-1, axis=0)
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    try:
        warped = cv2.warpPerspective(img, M, (width, height))
        # save cropped img to file
        curr_shape = np.array(warped.shape[:2])
        rescaled = cv2.resize(warped, dsize=curr_shape*3, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(cropped_fpath, rescaled)
    except Exception as e:
        print(e)
        print(cropped_fpath)
        print(img)
        print("shape of corners: {}".format(corners.shape))
        print("rect: {}".format(rect))
        print("bounding box: {}".format(box))


def plot_pfov(skycam_img, corners, pfov_fpath):
    """Plot the panoseti module FoV on the sky camera image"""
    # Blue color in BGR
    color = (255, 255, 255)
    #skycam_img = skycam_img.reshape((-1, 1, 2))
    #print(skycam_img)
    poly_img = cv2.polylines(skycam_img,
                             [corners],
                             isClosed=True,
                             color=color,
                             thickness=1)
    # Save the image
    cv2.imwrite(pfov_fpath, poly_img)

def get_img_path(original_fname, img_type, skycam_dir):
    original_sc2_dir, cropped_out_dir, pfov_out_dir = get_img_dirs(skycam_dir)
    if original_fname[-4:] != '.jpg':
        return None
    if img_type == 'original':
        return f'{original_sc2_dir}/{original_fname}'
    elif img_type == 'cropped':
        return f'{cropped_out_dir}/{original_fname[:-4]}_cropped.jpg'
    elif img_type == 'pfov':
        return f'{pfov_out_dir}/{original_fname[:-4]}_pfov.jpg'


pixel_data_file = 'skycam_pixels.json'
sc2_img_dir = 'SC2_imgs_2023-08-01'

corners_4x1x2 = get_corners()
init_dirs(sc2_img_dir)
original_sc2_dir, cropped_out_dir, pfov_out_dir = get_img_dirs(sc2_img_dir)

for original_fname in os.listdir(original_sc2_dir):
    # load the image
    if original_fname[-4:] != '.jpg':
        continue

    original_img = cv2.imread(get_img_path(original_fname, 'original', sc2_img_dir))
    cropped_fpath = get_img_path(original_fname, 'cropped', sc2_img_dir)
    pfov_fpath = get_img_path(original_fname, 'pfov', sc2_img_dir)

    crop_img(original_img, corners_4x1x2, cropped_fpath)
    plot_pfov(original_img, corners_4x1x2, pfov_fpath)

