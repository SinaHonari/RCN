"""
This module creates the 68 keypoint 300W datasets (as pickle files) by using the orginal images in 'src_dir'. It keeps the images in their
RGB format but with a reduced_size.

#################################

Here is the instruction to create 300W datasets:

1 - Download Helen, LFPW, AFW and IBUG datasets from:
http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

2 - Once unzipped, the helen and lfpw have two subdirectories, 'testset' and 'trainset'.
Rename them to 'X_testset' and 'X_trainset', for each dataset X.

3 - Create one directory named 'Train_set' and put unzipped 'afw', 'helen_trainset'
and 'lfpw_trainset' directories into it (as three sub-directories).

4 - Create another directory named 'Test_set' and put unzipped 'ibug', 'helen_testset' and 'lfpw_testset'
into it (as three sub-directories).

5 - Put Train_set and Test_set directories into one direcotory (i.e. 300W) and pass
the complete path to it to 'src_dir' when calling this module.

6 - Call create_raw_300W.py module by passing complete path to 'src_dir' and 'dest_dir' arguments:
python create_raw_300W.py --src_dir=/complete/path/to/300W/folder --dest_dir=/complete/path/to/RCN/datasets

**Note: dest_dir is the location where the dataset will be created. It should be finally put in RCN/datasets directory
of the repo

This module will create 300W_test_160by160.pickle and 300W_train_160by160.pickle files in the given dest_dir path.

**Note: follow the instructions here for citation if you use 300W dataset:
http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

#################################

Note: This module creates a dataset with 3148 train images (afw, Helen, lfpw), and 689 test images (135 ibug, 330 Helen, 224 lfpw)
with 68 keypoints.

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

The created datasets are in two files, MTFL_test and MTFL_train. Each file contains an orderedDict with 'X' and 'Y' label.
'X' is a 4D tensor of size (#sampels, #rows, #cols, #channels). Each image sample is in cv2's BGR format
'Y' is an orderedDict with the following components. Each conponent has as many rows as the number of samples.
'kpt_orig'    : the original position of the keypoints in the format
                x1 y1 ... x5 y5: the locations for left eye, right eye, nose, left mouth corner, right mouth corner.
'kpt_norm'    : the normalized keypoint positions such that each x or y is in the range of [0,1]
                key_point_x_normalized = ( keypoint_x - rect_start_x ) / rect_width
'face_rect'   : a four value vector of values: rect_start_x, rect_start_y, rect_width (rect_end_x - rect_start_x), rect_height (rect_end_y - rect_start_y)
"""

import cv2
import cv
import os
import numpy as np
from PIL import Image
from collections import OrderedDict
import cPickle as pickle
import string
import copy
import argparse

def box(rects, img, out_img):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 10)
    return img

reduced_size = (160, 160)

def crop_img(img, x_start, y_start, x_end, y_end):
    # the method for cropping an image
    return img.crop((x_start, y_start, x_end, y_end))

def resize_img(img, size):
    # the methods for downsampling
    return img.resize(size, Image.ANTIALIAS)

def get_kpts(file_path):
    kpts = []
    f = open(file_path, 'r')
    ln = f.readline()
    while not ln.startswith('n_points'):
        ln = f.readline()

    num_pts = ln.split(':')[1]
    num_pts = string.strip(num_pts)
    # checking for the number of keypoints
    if float(num_pts) != num_kpts:
        print "encountered file with less than %f keypoints in %s" %(num_kpts, file_pts)
        return None

    # skipping the line with '{'
    ln = f.readline()

    ln = f.readline()
    while not ln.startswith('}'):
        vals = ln.split(' ')[:2]
        vals = map(string.strip, vals)
        vals = map(np.float32, vals)
        kpts.append(vals)
        ln = f.readline()
    return kpts

def get_border_mask(kpts, img_width, img_height):
    # This module gets whether each keypoint touches the boundary,
    # for top and left boundaries if the values are in the range [0,10]
    # it is considered on the boundary
    # for bottom and right boundaries if they are within 3 pixels from the border
    # they are considered on the boundary
    # it returns a vector of size (#kpts), where for each keypoint
    # one of these 4 values is given:
    # 0: no_boundary, 1: left boundary, 2: top boundary
    # 3: right boundary, 4: bottom boundary
    flags = np.zeros(kpts.shape[0])

    # checking for left boundary
    left_bound = kpts[:,0] < 10
    flags += left_bound * 1

    # checking for top boundary
    top_bound = kpts[:,1] < 10
    flags += top_bound * 2

    # checking for right boundary
    right_bound = (img_width - kpts[:,0]) <= 3
    flags += right_bound * 3

    # checking for bottom boundary
    bottom_bound = (img_height - kpts[:,1]) <= 3
    flags += bottom_bound * 4
    return flags

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

def validate_bbox(x_start, x_end, y_start, y_end, img_width, img_height):
    x_start_extra = 0
    y_start_extra = 0
    x_end_extra = 0
    y_end_extra = 0
    if x_start < 0:
        x_start_extra = -x_start
    if y_start < 0:
        y_start_extra = -y_start
    if x_end > img_width:
        x_end_extra = x_end - img_width
    if y_end > img_height:
        y_end_extra = y_end - img_height

    max_margin = np.max((x_start_extra, y_start_extra, x_end_extra, y_end_extra))

    if max_margin > 0:
        x_start += max_margin
        y_start += max_margin
        x_end -= max_margin
        y_end -= max_margin

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
    return (rect_init_square, max_margin)

def enlarge_bbox(bbox, ratio=1.25):
    x_start, y_start, width, height = bbox
    x_end = x_start + width
    y_end = y_start + height
    assert width == height, "width %s is not equal to height %s" %(width, height)
    assert ratio > 1.0 , "ratio is not greater than one. ratio = %s" %(ration,)
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

def increase_img_size(img, bbox):
    # this method increases the bounding box size
    # if start and end values for the bounding box
    # go beyond the image size (from either side)
    # and in such a case gets the ratio of padded region
    # to the total image size (total_img_size = orig_size + pad)
    x_start, y_start, width_bbox, height_bbox = bbox
    x_end = x_start + width_bbox
    y_end = y_start + height_bbox
    height, width, _ = img.shape

    x_extra_start = 0
    x_extra_end = 0
    y_extra_start = 0
    y_extra_end = 0

    if x_start < 0:
        x_extra_start = - x_start
    if x_end > width:
        x_extra_end = x_end - width
    if y_start < 0:
        y_extra_start = - y_start
    if y_end > height:
        y_extra_end = y_end - height

    # extending the image
    img_new = copy.copy(img)
    x_extra = x_extra_start + x_extra_end
    if x_extra > 0:
        img_new = np.zeros((height, width + x_extra, 3))
        #img_new[ :, : x_extra_start, :] = colbs
        img_new[ :, x_extra_start : x_extra_start + width, :] = img
        #img_new[ :, x_extra_start + width :  x_extra_start + width + x_extra_end, :] = coles

    width = img_new.shape[1]
    y_extra = y_extra_start + y_extra_end
    img_new2 = copy.copy(img_new)
    if y_extra > 0:
        img_new2 = np.zeros((height + y_extra, width, 3))
        #img_new2[ : y_extra_start, :, :] = rowbs
        img_new2[y_extra_start : y_extra_start + height, :, :] = img_new
        #img_new2[y_extra_start + height :  y_extra_start + height + y_extra_end, :, :] = rowes

    # getting the ration of the padded region to the total image size
    total_width = float(width_bbox)
    left_ratio = x_extra_start / total_width
    right_ratio = x_extra_end / total_width
    total_height = float(height_bbox)
    top_ratio = y_extra_start / total_height
    bottom_ratio = y_extra_end / total_height
    pad_ratio = [left_ratio, top_ratio, right_ratio, bottom_ratio]

    # checking bounding box size after image padding
    height, width, _ = img_new2.shape
    if x_extra_start:
        x_start = 0
        x_end += x_extra_start
    if y_extra_start:
        y_start = 0
        y_end += y_extra_start
    if x_extra_end:
        x_end = width
    if y_extra_end:
        y_end = height
    bbox_width = x_end - x_start
    bbox_height = y_end - y_start
    assert bbox_width == bbox_height

    rect = [x_start, y_start, bbox_width, bbox_height]

    #import pdb; pdb.set_trace()
    if y_extra or x_extra:
        print "extended the image"
    return img_new2, rect, x_extra_start, y_extra_start, pad_ratio

def process_files(source_dir, folder, subset):
    X = []
    Y_kpt_orig = []
    Y_kpt = []
    Y_rect = []
    Y_bound_mask = []
    Y_pad_ratio = []
    Y_img_name = []

    path = "%s/%s" %(source_dir, folder)
    files = os.listdir(path)
    files = [i for i in files if i.endswith('.pts')]

    sanity = False # undownsampled image chacking
    check = False # downsampled image checking
    for index, file_pts in enumerate(files):
        file_path = "%s/%s" %(path, file_pts)
        kpts =  get_kpts(file_path)
        if kpts is None:
            continue

        file_jpg = file_pts.split('.')[0] + '.jpg'
        jpg_path =  "%s/%s" %(path, file_jpg)
        if not os.path.isfile(jpg_path):
            file_jpg = file_pts.split('.')[0] + '.png'
            jpg_path =  "%s/%s" %(path, file_jpg)
        Y_img_name.append(file_jpg)
        im = Image.open(jpg_path)
        img_width, img_height = im.size
        #print "width %s, height %s" %(img_width, img_height)

        # getting left eye information
        l_eye1_x, l_eye1_y = kpts[l_eye_1_indx]
        l_eye2_x, l_eye2_y = kpts[l_eye_2_indx]
        l_eye_x = np.mean((l_eye1_x, l_eye2_x))
        l_eye_y = np.mean((l_eye1_y, l_eye2_y))

        # getting right eye information
        r_eye1_x, r_eye1_y = kpts[r_eye_1_indx]
        r_eye2_x, r_eye2_y = kpts[r_eye_2_indx]
        r_eye_x = np.mean((r_eye1_x, r_eye2_x))
        r_eye_y = np.mean((r_eye1_y, r_eye2_y))

        # getting left eye information
        l_mouth_x, l_mouth_y = kpts[l_mouth_indx]
        r_mouth_x, r_mouth_y = kpts[r_mouth_indx]

        # checking if the bounding box covers the whole face
        kpts = np.array(kpts)
        x_min = np.min(kpts[:,0])
        x_max = np.max(kpts[:,0])
        y_min = np.min(kpts[:,1])
        y_max = np.max(kpts[:,1])

        # detect the initial bbox using only the min and max values of the keypoints
        bbox, max_margin = get_bbox(x_min, y_min, x_max, y_max)
        border_mask = get_border_mask(kpts, img_width, img_height)
        # enlarge the bbox by a ratio
        rect, max_margin = enlarge_bbox(bbox, ratio=1.7)

        x_start, y_start, width, height = rect
        x_end = x_start + width
        y_end = y_start + height

        assert x_start < x_min, "x_state %s is not smaller than x_min %s" %(x_start, x_min)
        assert x_end > x_max, "x_end %s is not bigger than x_max %s" %(x_end, x_max)
        assert y_start < y_min, "y_state %s is not smaller than y_min %s" %(y_start, y_min)
        assert y_end > y_max, "y_end %s is not bigger than y_max %s" %(y_end, y_max)

        ###################
        # sanity checking #
        ###################
        """
        Note: the sanity section (if set to true) writes to an output file the detected rectangle and the keypoint locations on them
              to verify the keypoint locations and the rectangle are correct.
        """
        if sanity:
            source_dir = parent_dir + '/created_datasets' + '/kpt_detection_raw'
            if not os.path.exists(source_dir):
                try:
                    os.makedirs(source_dir)
                except OSError:
                    if not os.path.isdir(source_dir):
                        raise

            sub_dir = source_dir + '/' + subset
            if not os.path.exists(sub_dir):
                try:
                    os.makedirs(sub_dir)
                except OSError:
                    if not os.path.isdir(sub_dir):
                        raise

            san_dir = sub_dir + '/' + folder
            if not os.path.exists(san_dir):
                try:
                    os.makedirs(san_dir)
                except OSError:
                    if not os.path.isdir(san_dir):
                        raise

            out_file_path = san_dir + '/' + file_jpg

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

            #increasing the image size
            img_npy, rect, x_extra_start, y_extra_start, pad_ratio = increase_img_size(img_npy, rect)
            x_start, y_start, width, height = rect
            x_end = x_start + width
            y_end = y_start + height
            img_height, img_width, _ = img_npy.shape

            if x_extra_start:
                kpts[:,0] += x_extra_start
            if y_extra_start:
                kpts[:,1] += y_extra_start

            for kpt in kpts:
                c, r = kpt[1], kpt[0]
                if r < img_width and c < img_height:
                    if len(img_npy.shape) == 3:
                        #img_npy[c,r] = (0,255,0)
                        cv2.circle(img_npy, (r,c), 2, (0,255,0))
                    else: # it has a len of 2
                        #img_npy[c,r] = 1.0 # filling with white value
                        cv2.circle(img_npy, (r,c), 2, 1.0)

            rects = [[x_start, y_start, x_end, y_end]]
            img_out = box(rects, img_npy, out_file_path)

            cv2.imwrite(out_file_path, img_out);
            #import pdb; pdb.set_trace()
        #######################
        # end sanity checking #
        #######################

        #########################################
        # getting the train and test datasets   #
        #########################################
        else:
            img = Image.open(jpg_path)
            #####################################
            # converting from PIL to cv2 format #
            #####################################
            # in order to convert PIL image to cv2 format, the image should be converted to RGB first
            img_rgb = img.convert('RGB')
            # getting the numpy array from image, this numpy array is in RGB format
            img_npy=np.asarray(img_rgb)
            # cv2 images are in BGR fortmat, so RGB should be changed to BGR
            if len(img_npy.shape) == 3:
                img_npy = img_npy[:, :, ::-1].copy()
            # if the image is in gray-scale convert it to BGR
            elif len(img_npy.shape) == 2:
                img_npy = cv2.cvtColor(img_npy, cv2.COLOR_GRAY2BGR)

            #increasing the image size
            img_npy, rect, x_extra_start, y_extra_start, pad_ratio = increase_img_size(img_npy, rect)
            x_start, y_start, width, height = rect
            x_end = x_start + width
            y_end = y_start + height
            img_height, img_width, _ = img_npy.shape

            if x_extra_start:
                kpts[:,0] += x_extra_start
            if y_extra_start:
                kpts[:,1] += y_extra_start

            #####################################
            # converting from cv2 to PIL format #
            #####################################
            #img = Image.fromstring("RGB", cv.GetSize(img_npy), img_npy.tostring())
            # converting from BGR to RGB
            img_RGB = img_npy[:, :, ::-1].copy()
            # making sure values are uint8
            img_RGB=np.uint8(img_RGB)
            img = Image.fromarray(img_RGB)

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
            # getting the normalized keypoints
            kpts_norm = np.zeros_like(kpts)
            kpts_norm[:,0] = kpts[:,0] - x_start
            kpts_norm[:,1] = kpts[:,1] - y_start
            kpts_norm[:,0] /= width
            kpts_norm[:,1] /= height
            #import pdb; pdb.set_trace()

            if check:
                kpts_checks = kpts_norm.copy()
                kpts_checks *= reduced_size[0]
                img_check = img_npy.copy()
                for kpt in kpts_checks:
                    c, r = kpt[1], kpt[0]
                    if r < img_width and c < img_height:
                        if len(img_check.shape) == 3:
                            img_check[c,r] = (0,255,0)
                            #cv2.circle(img_check, (r,c), 1, (0,255,0))
                        else: # it has a len of 2
                            img_check[c,r] = 1.0 # filling with white value
                            #cv2.circle(img_check, (r,c), 1, 1.0)

                source_dir = parent_dir + '/created_datasets' + '/kpt_detection_downsampled'
                if not os.path.exists(source_dir):
                    try:
                        os.makedirs(source_dir)
                    except OSError:
                        if not os.path.isdir(source_dir):
                            raise
                sub_dir = source_dir + '/' + subset
                if not os.path.exists(sub_dir):
                    try:
                        os.makedirs(sub_dir)
                    except OSError:
                        if not os.path.isdir(sub_dir):
                            raise
                san_dir = sub_dir + '/' + folder
                if not os.path.exists(san_dir):
                    try:
                        os.makedirs(san_dir)
                    except OSError:
                        if not os.path.isdir(san_dir):
                            raise

                out_file_path = san_dir + '/' + file_jpg
                cv2.imwrite(out_file_path, img_check);
            # end check

            X.append(img_npy)
            val1, val2 = kpts.shape
            kpts = kpts.reshape(val1 * val2)
            Y_kpt_orig.append(kpts)
            kpts_norm = kpts_norm.reshape(val1 * val2)
            Y_kpt.append(kpts_norm)
            Y_rect.append(rect)
            Y_bound_mask.append(border_mask)
            Y_pad_ratio.append(pad_ratio)
        print "done with index %i file %s" %(index, file_path)

    return [X, Y_kpt_orig, Y_kpt, Y_rect, Y_bound_mask, Y_pad_ratio, Y_img_name]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creating the pickle files for 300W datasets.')
    parser.add_argument('--src_dir', type=str, help='the complete path to the folder containing Train_set and Test_set folders', required=True)
    parser.add_argument('--dest_dir', type=str, help='the complete path to the folder where the created pickle files with be saved', required=True)

    args = parser.parse_args()
    num_kpts = 68
    # the indices for the important key points
    # remving one since the indices start from zero in python
    l_eye_1_indx = 37 - 1
    l_eye_2_indx = 40 - 1
    r_eye_1_indx = 43 - 1
    r_eye_2_indx = 46 - 1
    l_mouth_indx = 49 - 1
    r_mouth_indx = 55 - 1

    # the path to the downsampled directory
    parent_dir = args.src_dir
    out_dir = args.dest_dir

    train_dir = "%s/Train_set" %(parent_dir)
    test_dir = "%s/Test_set" %(parent_dir)

    X_train = []
    Y_train = OrderedDict()
    Y_train_kpt_orig = []      # the original position of the keypoints in the format
                               # x1 y1 ... x5 y5: the locations for left eye, right eye, nose, left mouth corner, right mouth corner.
    Y_train_kpt = []           # the normalized keypoint positions such that each x or y is in the range of [0,1]
                               # key_point_x_normalized = ( keypoint_x - rect_start_x ) / rect_width
                               # key_point_y_normalized = ( keypoint_y - rect_start_y ) / rect_height
    Y_train_rect = []          # a four value vector of values: rect_start_x, rect_start_y,
                               # rect_width (rect_end_x - rect_start_x), rect_height (rect_end_y - rect_start_y)
    Y_train_bound_mask = []    # an int value for each ketpoint indicating whether it is on the border or not:
                               # 0: no_boundary, 1: left boundary, 2: top boundary
                               # 3: right boundary, 4: bottom boundary
    Y_train_pad_ratio = []     # A four valued float vector per image indicating the ratio of padded region to the total_img_size(orig + padded)
                               # the pad_ratio is in this order: left, top, right, bottom
                               # Note that since these values go through a image size reduction, the ratio should be multiplied in the image size
                               # and then rounded to the closest interger value
    Y_train_name = []          # the name of the image file

    # goiing through the train set
    subdirs = os.listdir(train_dir)
    subdirs = [i for i in subdirs if not i.startswith('.')]
    for subdir in subdirs:
        x, y_kpt_orig, y_kpt, y_rect, y_bound_mask, y_pad_ratio, y_train_img_name = process_files(train_dir, subdir, 'Train_set')
        X_train.extend(x)
        Y_train_kpt_orig.extend(y_kpt_orig)
        Y_train_kpt.extend(y_kpt)
        Y_train_rect.extend(y_rect)
        Y_train_bound_mask.extend(y_bound_mask)
        Y_train_pad_ratio.extend(y_pad_ratio)
        Y_train_name.extend(y_train_img_name)

    X_train = np.array(X_train)
    Y_train['kpt_orig'] = np.float32(Y_train_kpt_orig)
    Y_train['kpt_norm'] = np.float32(Y_train_kpt)
    Y_train['face_rect'] = np.int16(Y_train_rect)
    Y_train['bound_mask'] = np.int8(Y_train_bound_mask)
    Y_train['pad_ratio'] = np.float32(Y_train_pad_ratio)
    Y_train['img_name'] = np.array(Y_train_name)
    # creating an OrderedDict for the train set
    train_set = OrderedDict()
    train_set['X'] = X_train
    train_set['Y'] = Y_train

    train_pkl_path = out_dir + '/300W_train_160by160.pickle'
    with open(train_pkl_path, 'wb') as fp:
        pickle.dump(train_set, fp)

    test_set = OrderedDict()

    # goiing through the test set
    subdirs = os.listdir(test_dir)
    subdirs = [i for i in subdirs if not i.startswith('.')]
    for subdir in subdirs:
        x, y_kpt_orig, y_kpt, y_rect, y_bound_mask, y_pad_ratio, y_img_name = process_files(test_dir, subdir, 'Test_set')
        x = np.array(x)
        y_kpt_orig = np.float32(y_kpt_orig)
        y_kpt = np.float32(y_kpt)
        y_rect = np.int16(y_rect)
        y_bound_mask = np.int8(y_bound_mask)
        y_pad_ratio = np.float32(y_pad_ratio)
        y_img_name = np.array(y_img_name)
        # creating an OrderedDict for the subset of test
        sub_test = OrderedDict()
        sub_test['X'] = x
        sub_test_y = OrderedDict()
        sub_test_y['kpt_orig'] = y_kpt_orig
        sub_test_y['kpt_norm'] = y_kpt
        sub_test_y['face_rect'] = y_rect
        sub_test_y['bound_mask'] = y_bound_mask
        sub_test_y['pad_ratio'] = y_pad_ratio
        sub_test_y['img_name'] = y_img_name
        sub_test['Y'] = sub_test_y
        # adding the created subset to the test orderedDict
        test_set[subdir] = sub_test

    test_pkl_path = out_dir + '/300W_test_160by160.pickle'
    with open(test_pkl_path, 'wb') as fp:
        pickle.dump(test_set, fp)
