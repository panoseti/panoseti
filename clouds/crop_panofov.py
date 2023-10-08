#! /usr/bin/env python3
# Crops the panoseti field of view in Lick all-sky camera images
# Assumes Lick webcam
#   - SC images are 480x640 pixels
#   - SC2 images are 521x765 pixels

import json
import numpy as np
import cv2
import os

from PIL import Image

pixel_data_file = 'sky_cam_pixels.json'
sc2_imgs = 'all_sky_matlab/SC2_imgs'
out_dir = 'cropped_imgs'

# Get pixel corner data created by panofovlickwebc.m
with open(pixel_data_file, 'r') as fp:
    pixel_data = json.load(fp)
    x_corners = np.round(pixel_data['SC2']['astrometry_entries'][0]['x_corners'])
    y_corners = np.round(pixel_data['SC2']['astrometry_entries'][0]['y_corners'])
    # Stack coords into an array of 2D points
    corners = np.vstack((x_corners, y_corners)).astype(np.int64).T
    corners = np.expand_dims(corners, axis=1)

for img_fname in os.listdir(sc2_imgs):
    # Transformation code from https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
    # load the image
    if img_fname[-4:] != '.jpg':
        continue
    img = cv2.imread(f'{sc2_imgs}/{img_fname}')

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

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    try:
        warped = cv2.warpPerspective(img, M, (width, height))
    except Exception:
        print(img_fname)
        print(img)
        print("shape of corners: {}".format(corners.shape))
        print("rect: {}".format(rect))
        print("bounding box: {}".format(box))

    # save cropped img to file
    cropped_img_fname = f'{out_dir}/{img_fname[:-4]}_cropped.jpg'
    cv2.imwrite(cropped_img_fname, warped)