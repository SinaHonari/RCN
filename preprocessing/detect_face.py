"""
This module detects a bounding box around a face using the keypoints,

The inputs are the eye and mouth keypoints and a distance ratio 'dist_ratio'. 
This is the ratio by which the max of (distance-between-eyes and the-distance-between-eye-and-mouth-centers)
is multiplied. Higher values indicate bigger bounding box and vice-versa.

It returns the detected bounding box values [x_start, y_start, x_end, y_end]
"""
import sys
import cv2
import os
import numpy as np
from PIL import Image

def detect_face_by_kpt(l_eye_x, l_eye_y, r_eye_x, r_eye_y, l_mouth_x, l_mouth_y, r_mouth_x, r_mouth_y, img_width, img_height, dist_ratio):
    """
    Parameters:
    -----------
        img_width: int
                  the width of the input image
        img_height: int
                    the height of the input image
        dist_ratio: float
                    the ratio by which the max of (eye-distance and eye-center-mouth-center-distance) should be multiplied in for face detection
    """
    #####################################
    # finding the rectangle around face #
    #####################################
    width = img_width
    height = img_height

    # getting the distance between the key-points
    eye_distance = np.sqrt((r_eye_x - l_eye_x)**2 + (r_eye_y - l_eye_y)**2)
    eye_ctr = [(r_eye_x + l_eye_x ) / 2.0 , (r_eye_y + l_eye_y ) / 2.0]
    mouth_ctr = [(r_mouth_x + l_mouth_x ) / 2.0 , (r_mouth_y + l_mouth_y ) / 2.0]
    mouth_eye_distance = np.sqrt((eye_ctr[0] - mouth_ctr[0])**2 + (eye_ctr[1] - mouth_ctr[1])**2)

    # getting the min and max x , y values of the key-points
    min_x = int(min(l_eye_x, r_eye_x, l_mouth_x, r_mouth_x))
    max_x = int(max(l_eye_x, r_eye_x, l_mouth_x, r_mouth_x))
    min_y = int(min(l_eye_y, r_eye_y, l_mouth_y, r_mouth_y))
    max_y = int(max(l_eye_y, r_eye_y, l_mouth_y, r_mouth_y))

    # increasing the smallest of x and y distance to the other one
    y_dist = max_y - min_y
    x_dist = max_x - min_x
    if y_dist < x_dist:
        min_y = min_y - np.ceil((x_dist - y_dist)/2.0)
        max_y = max_y + np.floor((x_dist - y_dist)/2.0)
    else:
        min_x = min_x - np.ceil((y_dist - x_dist)/2.0)
        max_x = max_x + np.floor((y_dist - x_dist)/2.0)

    y_dist = max_y - min_y
    x_dist = max_x - min_x
    assert y_dist == x_dist, "1 - y_dist %s, x_dist %s" %(y_dist, x_dist)
        
    # increasing the boundary to include the face
    distance = int(max(eye_distance, mouth_eye_distance) * dist_ratio)
    x_start = int(min_x - distance)
    x_end = int(max_x + distance)
    y_start = int(min_y - distance)
    y_end = int(max_y + distance)

    y_dist = y_end - y_start
    x_dist = x_end - x_start
    assert y_dist == x_dist, "2 - y_dist %s, x_dist %s" %(y_dist, x_dist)

    if x_start < 0:
        x_end += (-x_start)
        x_start = 0
    x_extra = 0
    y_extra = 0
    if x_end > width:
        x_extra = x_end - width 
        x_begin_margin = x_start
        if x_extra <= x_begin_margin:
            x_start -= x_extra
            x_extra = 0
        else:
            x_extra -= x_begin_margin
            x_start = 0
        x_end = width
        
    if y_start < 0:
        y_end += (-y_start) 
        y_start = 0
    if y_end > height:
        y_extra = y_end - height 
        y_begin_margin = y_start
        if y_extra <= y_begin_margin:
            y_start -= y_extra
            y_extra = 0
        else:
            y_extra -= y_begin_margin
            y_start = 0
        y_end = height

    if x_extra > 0 and y_extra > 0:
        min_margin = min(x_extra, y_extra)
        x_extra -= min_margin
        y_extra -= min_margin

    if x_extra > 0:
        y_start += int(np.ceil(x_extra/2.0))
        y_end -= int(np.floor(x_extra/2.0))
    if y_extra > 0:
        x_start += int(np.ceil(y_extra/2.0))
        x_end -= int(np.floor(y_extra/2.0))

    assert x_start >=0, "x_start is less than zero"
    assert y_start >=0, "y_start is less than zero"
    assert x_end <= width, "x_end is greater than width"
    assert y_end <= height, "y_end is greater than height"
    y_dist = y_end - y_start
    x_dist = x_end - x_start
    assert y_dist == x_dist, "3 - y_dist %s, x_dist %s" %(y_dist, x_dist)

    rects = [x_start, y_start, x_end - x_start, y_end - y_start]
    return rects
    #########################################
    # end finding the rectangle around face #
    #########################################

def get_bbox(x1, y1, x2, y2):
    # This module makes the bounding box sqaure by
    # increasing the lower of the bounding width and height
    x_start = int(np.floor(x1))
    x_end = int(np.ceil(x2))
    y_start = int(np.floor(y1))
    y_end = int(np.ceil(y2))

    width = np.ceil(x_end - x_start)
    height = np.ceil(y_end - y_start)

    if width < height:
        diff = height - width
        x_start -=(np.ceil(diff/2.0))
        x_end +=(np.floor(diff/2.0))
    elif width > height:
        diff = width - height
        y_start -=(np.ceil(diff/2.0))
        y_end +=(np.floor(diff/2.0))

    width = x_end - x_start
    height = y_end - y_start
    assert width == height
    rect_init_square = [int(x_start), int(y_start), int(width), int(height)]
    max_margin = 0
    return (rect_init_square, max_margin)

def enlarge_bbox(bbox, ratio=1.25):
    x_start, y_start, width, height = bbox
    x_end = x_start + width
    y_end = y_start + height
    assert width == height, "width %s is not equal to height %s" %(width, height)
    #assert ratio > 1.0 , "ratio is not greater than one. ratio = %s" %(ration,)
    change = ratio - 1.0
    shift = (change/2.0)*width
    x_start_new = int(np.floor(x_start - shift))
    x_end_new = int(np.ceil(x_end + shift))
    y_start_new = int(np.floor(y_start - shift))
    y_end_new = int(np.ceil(y_end + shift))

    # assertion for increase lenght
    width = int(x_end_new - x_start_new)
    height = int(y_end_new - y_start_new)
    assert height == width
    max_margin = 0
    rect_init_square = [x_start_new, y_start_new, width, height]
    return (rect_init_square , max_margin)

def detect_face_300W(kpts, img_size, dist_ratio):
    num_kpts = 68
    kpts = kpts.reshape(num_kpts, 2)
    x_min = np.min(kpts[:,0])
    x_max = np.max(kpts[:,0])
    y_min = np.min(kpts[:,1])
    y_max = np.max(kpts[:,1])

    # getting bounding box around face using only min and max keypoints
    img_width = img_height = img_size
    bbox, max_margin = get_bbox(x_min, y_min, x_max, y_max)

    # enlarge the bbox by a ratio
    rect, max_margin = enlarge_bbox(bbox, dist_ratio)
    x_start, y_start, width, height = rect
    x_end = x_start + width
    y_end = y_start + height

    assert x_start > 0
    assert y_start > 0
    assert x_end < img_width
    assert y_end < img_height
    return rect

def detect_dataset_faces(Y, img_size, dist_ratio):
    """
    This method detects the face using keypoints for the whole dataset and returns the detected face.

    Parameters
    ----------
        'Y' is an orderedDict with at least the following components. Each component has as many rows as the number of samples.
                'kpt_norm'    : the normalized keypoint positions such that each x or y is in the range of [0,1]
                                key_point_x_normalized = ( keypoint_x - rect_start_x ) / rect_width
        img_width: int
                  the height or width of the input image

        dist_ratio: float
                    the ratio by which the max of (eye-distance and eye-center-mouth-center-distance) should be multiplied in for face detection

    returns:
    --------
        'rect'   : a matrix of size (#sample, four) corresponding to the detected rectangle for each sample
                   the four values are rect_start_x, rect_start_y, rect_width (rect_end_x - rect_start_x), rect_height (rect_end_y - rect_start_y)
    """
    kpts_norm = Y['kpt_norm']
    kpts_imgs = kpts_norm * img_size
    rects = []
    num_vals_MTFL = 10
    num_vals_300W = 136
    #MTFL dataset
    num_kpts = kpts_imgs.shape[1]
    if num_kpts == num_vals_MTFL:
        for kpts in kpts_imgs:
            l_eye_x, l_eye_y, r_eye_x, r_eye_y, nose_x, nose_y, l_mouth_x, l_mouth_y, r_mouth_x, r_mouth_y = kpts
            rect = detect_face_by_kpt(l_eye_x, l_eye_y, r_eye_x, r_eye_y, l_mouth_x, l_mouth_y, r_mouth_x, r_mouth_y, img_size, img_size, dist_ratio)
            rects.append(rect)
    elif num_kpts == num_vals_300W:
        for kpts in kpts_imgs:
            rect = detect_face_300W(kpts, img_size, dist_ratio)
            rects.append(rect)
    else:
        sys.stderr.write("the number of keypoints %i is not handled\n" %num_kpts)
        sys.exit(0)
        
    return np.array(rects)
