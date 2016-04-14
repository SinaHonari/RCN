"""
This module create the 5-keypoint afw test set pickle file by using the orginal/raw images given by 'src_dir'.
It keeps the images in their RGB format but with a reduced_size. This is one of the test sets with 5 keypoints. 
The other test set with 5 keypoints (AFLW) is created as part of MTFL dataset creation (see create_raw_MTFL.py
in the same directory for further instructions).

#################################

Here is the instruction to create 5-keypoint afw dataset:

1 - Download the images from:
https://www.ics.uci.edu/~xzhu/face/AFW.zip

2 - Unzip the folder and pass the complete path to it to 'src_dir' when calling create_raw_MTFL.py module.

3 - Call create_raw_MTFL.py module by passing complete path to 'src_dir' and 'dest_dir' arguments:
python create_raw_afw.py  --src_dir=/complete/path/to/AFW/unzipped/folder --dest_dir=/complete/path/to/RCN/datasets

**Note: dest_dir is the location where the dataset will be created. It should be finally put in RCN/datasets directory
of the repo

This module will create AFW_test_160by160.pickle in the given dest_dir path.

**Note: follow the instructions here for citation if you use AFW dataset:
https://www.ics.uci.edu/~xzhu/face/

#################################

The process in this module is as follows:
For all images in the train and test sets, a rectangle around face is detected.
the face is cropped within the detected rectangle and then downsampled to reduced_size.
In order to avoid aliasing while downsampling, Image library with Image.ANTIALIAS downsampling feature is used.
The downsampled Image is then converted from PIL to BGR format, which is the default format in cv2 images.
Then the original keypoint locations, the detected rectangle, and the normalized key-point locations
in the range [0, 1] is kept. Note that the normalized keypoint locations can be multiplied in the reduced_size (the image size) to get
the pixel locations.

All other pre-processings (detection of the bounding box, gray-scaling, downsampling, and contrast normalization)
are post-poned to later stages.

The created dataset is in AFW_test. Each file contains an orderedDict with 'X' and 'Y' label.

'X' is a 4D tensor of size (#sampels, #rows, #cols, #channels). Each image sample is in cv2's BGR format. rows and cols are the same
                                                                and correspond to the reduced size.
'Y' is an orderedDict with the following components. Each conponent has as many rows as the number of samples.

'name'                  : the name of the image.
'kpt_init'              : the initial position as provided in the datasets. The keypoints are in the format
                            x1 y1 ... x5 y5 x6 y6: the locations for left eye, right eye, nose, left mouth corner, center mouth, right mouth corner.
'kpt_orig'              : the original position of the keypoints without the center mouth keypoint. They come in the format
                            x1 y1 ... x5 y5: the locations for left eye, right eye, nose, left mouth corner, right mouth corner.
'kpt_norm'              : the normalized keypoint positions such that each x or y is in the range of [0,1] for the 5 keypoints in the kpt_orig
                            key_point_x_normalized = ( keypoint_x - rect_start_x ) / rect_width
'face_rect_init'        : a four value vector as provided in the dataset. The values are
                            [upper_left_corner x, upper_left_corner y, width, height]
'face_rect_init_square' : a four value vector close to the ones provided by the dataset. The values are
                            [upper_left_corner x, upper_left_corner y, width, height]
                            The difference with 'face_rect_init' is that the width and height are equal.
                            So the smaller of width and height is increased to match the other one.
'face_rect'             : a four value vector of values: rect_start_x, rect_start_y,
                            rect_width (rect_end_x - rect_start_x), rect_height (rect_end_y - rect_start_y)
                            detected using the keypoints.
'pose'                  : the pose of the face with values yaw, pitch, roll
"""

import cv2
import cv
import os
import numpy as np
from PIL import Image
from collections import OrderedDict
import cPickle as pickle
import argparse
from RCN.preprocessing.detect_face import detect_face_by_kpt
import scipy.io
from h5py import File
import h5py

def box(rects, img, out_img):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
    return img

def get_bbox(x1, y1, x2, y2, img_width, img_height):
    # This module makes the bounding box sqaure by 
    # increasing the lower of the bounding width and height
    width = x2 - x1
    height = y2 - y1
    x_start = x1
    x_end = x2
    y_start = y1
    y_end = y2

    if width < height:
        diff = height - width
        x_start -=(np.ceil(diff/2.0))
        x_end +=(np.floor(diff/2.0))
    elif width > height:
        diff = width - height
        y_start -=(np.ceil(diff/2.0))
        y_end +=(np.floor(diff/2.0))

    if x_start < 0:
        x_end += (-x_start)
        x_start = 0
    x_extra = 0
    y_extra = 0
    if x_end > img_width:
        x_extra = x_end - img_width 
        x_begin_margin = x_start
        if x_extra <= x_begin_margin:
            x_start -= x_extra
            x_extra = 0
        else:
            x_extra -= x_begin_margin
            x_start = 0
        x_end = img_width
        
    if y_start < 0:
        y_end += (-y_start) 
        y_start = 0
    if y_end > img_height:
        y_extra = y_end - img_height 
        y_begin_margin = y_start
        if y_extra <= y_begin_margin:
            y_start -= y_extra
            y_extra = 0
        else:
            y_extra -= y_begin_margin
            y_start = 0
        y_end = img_height

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

    #import pdb; pdb.set_trace()
    width2 = int(x_end - x_start)
    height2 = int(y_end - y_start)
    x_start = int(x_start)
    y_start = int(y_start)
    x_end = int(x_end)
    y_end = int(y_end)
    assert x_start >= 0
    assert y_start >= 0
    assert x_end <= img_width
    assert y_end <= img_height
    assert height2 == width2
    rect_init_square = [x_start, y_start, width2, height2]
    return rect_init_square
    

def crop_img(img, x_start, y_start, x_end, y_end):
    # the method for cropping an image
    return img.crop((x_start, y_start, x_end, y_end))

def resize_img(img, size):
    # the methods for downsampling
    return img.resize(size, Image.ANTIALIAS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creating AFW test set.')
    parser.add_argument('--src_dir', type=str, help='the complete path to the folder containing AFW images', required=True)
    parser.add_argument('--dest_dir', type=str, help='the complete path to the folder where the created pickle files with be saved', required=True)    
    parser.add_argument('--extended', action='store_true', default=False)
    args = parser.parse_args()
    # If extended is False, the images are taken in a more closed-up format.
    # If extended is True, the images are taken in zoomed-out format to avoid having
    # black regions after rotating the image. However, less detail on faces is captured
    # since the resolution remains the same. The extended is mostly used to train
    # the rotation model, where the facial details are not important.
    extended = args.extended
    print "extended flag is %s " %(extended,)

    reduced_size = (160, 160)

    parent_dir = args.src_dir
    out_dir = args.dest_dir

    #reading testing file
    testing = parent_dir + '/anno.mat'
    f = h5py.File(testing)
    f_base = f[u'anno']

    X = []                     # the reduced size of the croped face
    Y_name = []                # the name of the image file
    Y_kpt_init = []            # the 6 kpts given in the dataset
    Y_kpt_orig = []            # the 5 keypoints without the mouth-center
    Y_kpt = []                 # the normalized keypoint position in the range [0,1]
    Y_rect_init = []           # the face bounding box provided in the dataset which is saved in format [x1, y1, width, height]
    Y_rect_init_square = []    # the face bounding box provided in the dataset but made square by increasing the lower of width and height
                               # which is saved in format [x1, y1, width, height]
    Y_rect = []                # the face bounding box detected by keypoints which is saved in format [x1, y1, width, height]
    Y_pose = []                # the face pose provided by the dataset

    sanity = False
    for indx in xrange(len(f_base[0])):
        for face_indx in xrange(len(f[f[u'anno'][1][indx]])):
            # getting the name of the image
            obj = f[f_base[0][indx]]
            name = ''.join(chr(i) for i in obj[:])
            
            #getting the bounding boxes of the face [x1, y1, x2, y2]. (upper left corner and lower right corner)
            obj = f[f[f_base[1][indx]][face_indx][0]]
            x1 = int(round(obj[0,0]))
            y1 = int(round(obj[1,0]))
            x2 = int(round(obj[0,1]))
            y2 = int(round(obj[1,1]))
            width = x2 - x1
            height = y2 - y1
            rect_init = [x1, y1, width, height]
            
            # getting pose [yaw, pitch, roll]
            obj = f[f[f_base[2][indx]][face_indx][0]]
            yaw = float(obj[0])
            pitch = float(obj[1])
            roll = float(obj[2])

            # getting 6 landmarks. (left eye, right eye, nose, left mouth, mouth center, mouth right)
            obj = f[f[f_base[3][indx]][face_indx][0]]
            l_eye_x = float(obj[0,0])
            r_eye_x = float(obj[0,1])
            nose_x = float(obj[0,2])
            l_mouth_x = float(obj[0,3])
            c_mouth_x = float(obj[0,4])
            r_mouth_x = float(obj[0,5])

            l_eye_y = float(obj[1,0])
            r_eye_y = float(obj[1,1])
            nose_y = float(obj[1,2])
            l_mouth_y = float(obj[1,3])
            c_mouth_y = float(obj[1,4])
            r_mouth_y = float(obj[1,5])

            kpts = [l_eye_x, l_eye_y, r_eye_x, r_eye_y, nose_x, nose_y, l_mouth_x, l_mouth_y, c_mouth_x, c_mouth_y, r_mouth_x, r_mouth_y]

            # checking if any of the keypoints is nan
            if np.isnan(kpts).any():
                continue
            Y_name.append(name)
            Y_rect_init.append(rect_init)
            Y_pose.append([yaw, pitch, roll])
            Y_kpt_init.append([l_eye_x, l_eye_y, r_eye_x, r_eye_y, nose_x, nose_y, l_mouth_x, l_mouth_y, c_mouth_x, c_mouth_y, r_mouth_x, r_mouth_y])
            Y_kpt_orig.append([l_eye_x, l_eye_y, r_eye_x, r_eye_y, nose_x, nose_y, l_mouth_x, l_mouth_y, r_mouth_x, r_mouth_y])
            file_path = parent_dir + '/' + name

            im = Image.open(file_path)
            width, height = im.size

            rect_init_square = get_bbox(x1, y1, x2, y2, width, height)
            Y_rect_init_square.append(rect_init_square)

            ##############################################
            # face detection for a bigger rectangle area #
            ##############################################
            print name
            if extended:
                dist_ratio = 2.5
            else:
                dist_ratio = 1.25
            rect = detect_face_by_kpt(l_eye_x, l_eye_y, r_eye_x, r_eye_y, l_mouth_x, l_mouth_y, r_mouth_x, r_mouth_y, img_width=width, img_height=height,
                                      dist_ratio=dist_ratio)
            rectangle = rect

            x_start, y_start, width, height = rect
            x_end = x_start + width
            y_end = y_start + height

            ###################
            # sanity checking #
            ###################
            """
            Note: the sanity section (if set to true) writes to an output file the detected rectangle and the keypoint locations on them
                  to verify the keypoint locations and the rectangle are correct.
            """
            if sanity:
                source_dir = parent_dir + '/Preprocessings/face_detection/keypoint_face_detect' + '/kpt_detection_raw_afw'
                if not os.path.exists(source_dir):
                    try: 
                        os.makedirs(source_dir)
                    except OSError:
                        if not os.path.isdir(source_dir):
                            raise

                san_dir = source_dir + '/' + folder
                if not os.path.exists(san_dir):
                    try:
                        os.makedirs(san_dir)
                    except OSError:
                        if not os.path.isdir(san_dir):
                            raise

                out_file_path = san_dir + '/' + file_jpg

                kpts = [[l_eye_x, l_eye_y], [r_eye_x, r_eye_y], [nose_x, nose_y], [l_mouth_x, l_mouth_y], [r_mouth_x, r_mouth_y]]

                #####################################
                # converting from PIL to cv2 format #
                #####################################
                # in order to convert PIL image to cv2 format, the image should be converted to RGB first
                img_rgb = im.convert('RGB')
                # getting the numpy array from image, this numpy array is in RGB format
                img_npy=np.asarray(img_rgb)
                # cv2 images are in BGR fortmat, so RGB should be changed to BGR
                if len(img_npy.shape) == 3:
                    img_npy = img_npy[:, :, ::-1].copy() 
                # if the image is in gray-scale convert it to BGR
                elif len(img_npy.shape) == 2:
                    img_npy = cv2.cvtColor(img_npy, cv2.COLOR_GRAY2BGR)

                for kpt in kpts:
                    c, r = kpt[1], kpt[0]
                    if c < width and r < height:
                        if len(img_npy.shape) == 3:
                            img_npy[c,r] = (0,255,0)
                        else: # it has a len of 2
                            img_npy[c,r] = 1.0 # filling with white value

                rects = [[x_start, y_start, x_end, y_end]]
                img_out = box(rects, img_npy, out_file_path)

                cv2.imwrite(out_file_path, img_out);
            #######################
            # end sanity checking #
            #######################

            #########################################
            # getting the train and test datasets   #
            #########################################
            img = Image.open(file_path)

            # cropping the image
            cropped_img = crop_img(img, x_start, y_start, x_end, y_end)

            # downsampling image
            resized_img = resize_img(cropped_img, size=reduced_size)

            #####################################
            # converting from PIL to cv2 format #
            #####################################
            # in order to convert PIL image to cv2 format, the image should be converted to RGB first
            img_rgb = resized_img.convert('RGB')
            # getting the numpy array from image, this numpy array is in RGB format
            img_npy=np.asarray(img_rgb)
            # cv2 images are in BGR fortmat, so RGB should be changed to BGR
            if len(img_npy.shape) == 3:
                img_npy = img_npy[:, :, ::-1].copy() 
            # if the image is in gray-scale convert it to BGR
            elif len(img_npy.shape) == 2:
                img_npy = cv2.cvtColor(img_npy, cv2.COLOR_GRAY2BGR)

            # normalizing the keypoint locations after cropping and downsampling
            # to a range of [0,1]
            width = height = x_end - x_start
            l_eye_x_norm = (l_eye_x - x_start) / width
            r_eye_x_norm = (r_eye_x - x_start) / width
            nose_x_norm = (nose_x - x_start) / width
            l_mouth_x_norm = (l_mouth_x - x_start) / width
            r_mouth_x_norm = (r_mouth_x - x_start) / width
            l_eye_y_norm = (l_eye_y - y_start) / height
            r_eye_y_norm = (r_eye_y - y_start) / height
            nose_y_norm = (nose_y - y_start) / height
            l_mouth_y_norm = (l_mouth_y - y_start) / height
            r_mouth_y_norm = (r_mouth_y - y_start) / height

            X.append(img_npy)
            Y_kpt.append([l_eye_x_norm, l_eye_y_norm, r_eye_x_norm, r_eye_y_norm, nose_x_norm, nose_y_norm, l_mouth_x_norm, l_mouth_y_norm, r_mouth_x_norm, r_mouth_y_norm])
            Y_rect.append(rect)
            print "done with face #%i" %face_indx
            #import pdb; pdb.set_trace()
        print "done with file %s" %file_path

    X_test = np.array(X)
    Y_test_name = np.array(Y_name)
    Y_test_kpt_init = np.array(Y_kpt_init)
    Y_test_kpt_orig = np.array(Y_kpt_orig)
    Y_test_kpt = np.array(Y_kpt)
    Y_test_rect_init = np.array(Y_rect_init)
    Y_test_rect_init_square = np.array(Y_rect_init_square)
    Y_test_rect = np.array(Y_rect)
    Y_test_pose = np.array(Y_pose)

    X_test = np.float32(X_test)
    Y_test = OrderedDict()
    Y_test['name'] = Y_test_name
    Y_test['kpt_init'] = np.float32(Y_test_kpt_init)
    Y_test['kpt_orig'] = np.float32(Y_test_kpt_orig)
    Y_test['kpt_norm'] = np.float32(Y_test_kpt)
    Y_test['face_rect_init'] = np.int16(Y_test_rect_init)
    Y_test['face_rect_init_sqaure'] = np.int16(Y_test_rect_init_square)
    Y_test['face_rect'] = np.int16(Y_test_rect)
    Y_test['pose'] = np.float16(Y_test_pose)

    test_set = OrderedDict()
    test_set['X'] = X_test
    test_set['Y'] = Y_test

    nans = []
    nans_eye = []
    for row in Y_test_kpt_init:
        if np.isnan(row).any():
            nans.append(row)

        eye = row[0:4]
        if np.isnan(eye).any():
            nans_eye.append(row)

    suffix = ''
    if extended:
        suffix = '_extended'

    test_name = 'AFW_test_160by160'
    test_pkl_path = "%s/%s%s.pickle" %(out_dir, test_name, suffix)
    with open(test_pkl_path, 'wb') as fp:
        pickle.dump(test_set, fp)
