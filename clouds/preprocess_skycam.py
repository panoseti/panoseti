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

def init_dirs(skycam_dir):
    out_dir = f'processed_{skycam_dir}'
    cropped_img_out_dir = f'{out_dir}/cropped'
    pfov_out_dir = f'{out_dir}/pfov'
    for dir_name in [out_dir, cropped_img_out_dir, pfov_out_dir]:
        os.makedirs(dir_name, exist_ok=True)
    return out_dir, cropped_img_out_dir, pfov_out_dir


def get_corners():
    # Get pixel corner data created by panofovlickwebc.m
    with open(pixel_data_file, 'r') as fp:
        pixel_data = json.load(fp)
        x_corners = np.round(pixel_data['SC2']['astrometry_entries'][0]['x_corners'])
        y_corners = np.round(pixel_data['SC2']['astrometry_entries'][0]['y_corners'])
        # Stack coords into an array of 2D points
        # Points should be in the following order: in top left, top right, bottom right, bottom left
        corners = np.vstack((x_corners, y_corners)).astype(np.int64).T
        #print(corners.shape)
        return corners


def crop_img(img, corners, cropped_fname):
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

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    try:
        warped = cv2.warpPerspective(img, M, (width, height))
        # save cropped img to file

        cv2.imwrite(cropped_fname, warped)
    except Exception:
        print(img_fname)
        print(img)
        print("shape of corners: {}".format(corners.shape))
        print("rect: {}".format(rect))
        print("bounding box: {}".format(box))


def plot_pfov(img, corners, pfov_fname):
    # Blue color in BGR
    color = (255, 0, 0)
    poly_img = cv2.polylines(img.astype(np.int32), corners, isClosed=True, color=color, thickness=10)
    # Displaying the image
    cv2.imwrite(pfov_fname, poly_img)
    # poly = Polygon(corners, closed=False, fill=True)
    # ax = plt.imshow(img)
    # #fig, ax = plt.subplots()
    # ax.add_patch(poly)
    #
    # #x, y = poly.exterior.
    # #plt.plot(x, y, c="red")
    # plt.show()


pixel_data_file = 'skycam_pixels.json'
sc2_imgs = 'SC2_imgs_2023-08-15'

corners_4x2 = get_corners()
corners_4x1x2 = np.expand_dims(corners_4x2, axis=1)
out_dir, cropped_img_out_dir, pfov_out_dir = init_dirs(sc2_imgs)

for img_fname in os.listdir(sc2_imgs)[:10]:
    # load the image
    if img_fname[-4:] != '.jpg':
        continue

    cropped_fname = f'{cropped_img_out_dir}/{img_fname[:-4]}_cropped.jpg'
    pfov_fname = f'{pfov_out_dir}/{img_fname[:-4]}_pfov.jpg'

    img = cv2.imread(f'{sc2_imgs}/{img_fname}')
    crop_img(img, corners_4x1x2, cropped_fname)
    plot_pfov(img, corners_4x1x2, pfov_fname)

