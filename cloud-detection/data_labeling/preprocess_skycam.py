#! /usr/bin/env python3
# Crops the panoseti field of view in Lick all-sky camera images
# Assumes Lick webcam
#   - SC images are 480x640 pixels
#   - SC2 images are 521x765 pixels

import json
import numpy as np
import cv2
import os
import traceback

from fetch_skycam_imgs import download_night_skycam_imgs
from skycam_utils import get_skycam_dir, get_skycam_subdirs, get_skycam_img_path, init_preprocessing_dirs, is_data_preprocessed, is_data_downloaded


pixel_data_file = 'skycam_pixels.json'

def get_corners(skycam_type):
    """ Read the most recent pixel corner data created by panofovlickwebc.m"""
    # TODO: search for the last astrometry calibration in the file.
    with open(pixel_data_file, 'r') as fp:
        pixel_data = json.load(fp)
        x_corners = np.round(pixel_data[skycam_type]['astrometry_entries'][0]['x_corners'])
        y_corners = np.round(pixel_data[skycam_type]['astrometry_entries'][0]['y_corners'])
        # Stack coords into an array of 2D points
        corners_4x2 = np.vstack((y_corners, x_corners)).astype(np.int64).T
        corners_4x1x2 = np.expand_dims(corners_4x2, axis=1)     # Shape for CV2 coord arrays
        return corners_4x1x2


def crop_img(img, corners, cropped_fpath):
    """Extract the panoseti FoV from a Lick all-sky camera image."""
    # Transformation code adapted from https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
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
    color = (255, 255, 255)
    poly_img = cv2.polylines(skycam_img,
                             [corners],
                             isClosed=True,
                             color=color,
                             thickness=1)
    cv2.imwrite(pfov_fpath, poly_img)



def preprocess_skycam_imgs(skycam_type, year, month, day, first_t, last_t, root, verbose=False):
    """Run all preprocessing routines on the """
    skycam_dir = get_skycam_dir(skycam_type, year, month, day)
    skycam_path = f'{root}/{skycam_dir}'
    try:
        init_preprocessing_dirs(skycam_path)
    except FileExistsError as fee:
        print(fee)
        return None
    
    try:
        retval = download_night_skycam_imgs(skycam_type, year, month, day, first_t, last_t, verbose=verbose, root=root)
        if retval is None:
            return None
    except FileExistsError as fee:
        print(fee)

    if verbose: print('Running pre-processing routines.')

    corners_4x1x2 = get_corners(skycam_type)
    img_subdirs = get_skycam_subdirs(skycam_path)

    for original_fname in os.listdir(img_subdirs['original']):
        # load the image
        if original_fname[-4:] != '.jpg':
            continue

        original_img = cv2.imread(get_skycam_img_path(original_fname, 'original', skycam_path))
        cropped_fpath = get_skycam_img_path(original_fname, 'cropped', skycam_path)
        pfov_fpath = get_skycam_img_path(original_fname, 'pfov', skycam_path)

        crop_img(original_img, corners_4x1x2, cropped_fpath)
        plot_pfov(original_img, corners_4x1x2, pfov_fpath)


if __name__ == '__main__':
    preprocess_skycam_imgs('SC2', 2023, 8, 7, verbose=True)
    