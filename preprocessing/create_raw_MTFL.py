"""
This module creates the 5-keypoint MTFL dataset (as pickle files) by using the orginal/raw images in 'src_dir'.
It keeps the images in their RGB format but with a reduced_size. This module creates only AFLW test set.
This is one of the test sets with 5 keypoints. The other test set with 5 keypoints (AFW) is created separately by 
another module. (see create_raw_afw.py in the same directory for how to create it).
 
#################################

Here is the instruction to create 5-keypoint MTFL dataset:

1 - Download the images from:
http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip

2 - Unzip the folder and pass the complete path to it to 'src_dir' when calling create_raw_MTFL.py module.

3 - Call create_raw_MTFL.py by passing complete path to 'src_dir' and 'dest_dir' arguments:
python create_raw_MTFL.py  --src_dir=/complete/path/to/MTFL/unzipped/folder --dest_dir=/complete/path/to/RCN/datasets

**Note: dest_dir is the location where the dataset will be created. It should be finally put in RCN/datasets directory
of the repo

This module will create MTFL_test_160by160.pickle and MTFL_train_160by160.pickle in the given dest_dir path.

**Note: follow the instructions here for citation if you use MTFL dataset:
http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html
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

The created datasets are in two files, MTFL_test and MTFL_train. Each file contains an orderedDict with 'X' and 'Y' label.
'X' is a 4D tensor of size (#sampels, #rows, #cols, #channels). Each image sample is in cv2's BGR format
'Y' is an orderedDict with the following components. Each conponent has as many rows as the number of samples.

'name'        : the name of the image.
'kpt_orig'    : the original position of the keypoints in the format 
                x1 y1 ... x5 y5: the locations for left eye, right eye, nose, left mouth corner, right mouth corner.
'kpt_norm'    : the normalized keypoint positions such that each x or y is in the range of [0,1]
                key_point_x_normalized = ( keypoint_x - rect_start_x ) / rect_width
'face_rect'   : a four value vector of values: rect_start_x, rect_start_y, rect_width (rect_end_x - rect_start_x), rect_height (rect_end_y - rect_start_y)
'glasses'     : [0, 1] indicating whether the person is wearing glasses or not 
                glasses: 0 for wearing glasses, 1 for not wearing glasses.
'gender'      : [0, 1] indicating whether the gender of the person 
                gender: 0 for male, 1 for female
'smile'       : [0, 1] indicating whether the person is smiling or not 
                smile: 0 for smiling, 1 for not smiling
'pose'        : one of the [0,..,4] values indicating the pose of the head
                head pose: 0 for left profile, 1 for left, 2 for frontal, 3 for right, 4 for right profile
"""

import cv2
import cv
import os
import numpy as np
from PIL import Image
from collections import OrderedDict
import argparse
import cPickle as pickle
from RCN.preprocessing.detect_face import detect_face_by_kpt

def box(rects, img, out_img):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
    return img

reduced_size = (160, 160)

def crop_img(img, x_start, y_start, x_end, y_end):
    # the method for cropping an image
    return img.crop((x_start, y_start, x_end, y_end))

def resize_img(img, size):
    # the methods for downsampling
    return img.resize(size, Image.ANTIALIAS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creating MTFL dataset.')
    parser.add_argument('--src_dir', type=str, help='the complete path to the MTFL unzipped folder', required=True)
    parser.add_argument('--dest_dir', type=str, help='the complete path to the folder where the created pickle files with be saved', required=True)
    parser.add_argument('--extended', action='store_true', default=False)
    args = parser.parse_args()
    parent_dir = args.src_dir
    out_dir = args.dest_dir
    # If extended is False, the images are taken in a more closed-up format.
    # If extended is True, the images are taken in zoomed-out format to avoid having
    # black regions after rotating the image. However, less detail on faces is captured
    # since the resolution remains the same. The extended is mostly used to train
    # the rotation model, where the facial details are not important.
    extended = args.extended
    print "extended flag is %s " %(extended,)

    #reading training file
    training = parent_dir + '/training.txt'
    f_train = open( training, 'r')

    #reading testing file
    testing = parent_dir + '/testing.txt'
    f_test = open(testing, 'r')

    X_train = []
    Y_train_name = []     # the name of the image file
    Y_train_kpt_orig = [] # the original position of the keypoints in the format
                          # x1 y1 ... x5 y5: the locations for left eye, right eye, nose, left mouth corner, right mouth corner.
    Y_train_kpt = []  # the normalized keypoint positions such that each x or y is in the range of [0,1]
                      # key_point_x_normalized = ( keypoint_x - rect_start_x ) / rect_width
                      # key_point_y_normalized = ( keypoint_y - rect_start_y ) / rect_height
    Y_train_rect = [] # a four value vector of values: rect_start_x, rect_start_y, rect_width (rect_end_x - rect_start_x), rect_height (rect_end_y - rect_start_y)
    Y_train_gl = []   # [0, 1] indicating whether the person is wearing glasses or not 
                      # glasses: 0 for wearing glasses, 1 for not wearing glasses.
    Y_train_gen = []  # [0, 1] indicating whether the gender of the person 
                      # gender: 0 for male, 1 for female
    Y_train_sm = []   # [0, 1] indicating whether the person is smiling or not 
                      # smile: 0 for smiling, 1 for not smiling
    Y_train_pose = [] # one of the [0,..,4] values indicating the pose of the head
                      # head pose: 0 for left profile, 1 for left, 2 for frontal, 3 for right, 4 for right profile
     
    X_test = []
    Y_test_name = []
    Y_test_kpt_orig = []
    Y_test_kpt = []
    Y_test_rect = []
    Y_test_gl = []
    Y_test_gen = []
    Y_test_sm = []
    Y_test_pose = []

    sanity = False

    for f in [f_train, f_test]:
        if f == f_train:
            X = X_train
            Y_name = Y_train_name
            Y_kpt_orig = Y_train_kpt_orig
            Y_kpt = Y_train_kpt   
            Y_rect = Y_train_rect
            Y_gl = Y_train_gl 
            Y_gen = Y_train_gen 
            Y_sm = Y_train_sm  
            Y_pose = Y_train_pose
        else:
            X = X_test
            Y_name = Y_test_name
            Y_kpt_orig = Y_test_kpt_orig
            Y_kpt = Y_test_kpt
            Y_rect = Y_test_rect
            Y_gl = Y_test_gl 
            Y_gen = Y_test_gen 
            Y_sm = Y_test_sm  
            Y_pose = Y_test_pose

        for line in f:
            if line == ' ':
                break
            face_parts = line.split('/')
            if len(face_parts) < 2:
                face_parts = line.split('\\')
            folder = face_parts[0].strip()
        
            face_parts = face_parts[1].split(' ')
            file_jpg = face_parts[0]
            file_path = parent_dir + '/' + folder + '/' + file_jpg

            im = Image.open(file_path)
            width, height = im.size

            # getting the values of the key-points
            l_eye_x = float(face_parts[1])
            r_eye_x = float(face_parts[2])
            nose_x = float(face_parts[3])
            l_mouth_x = float(face_parts[4])
            r_mouth_x = float(face_parts[5])

            l_eye_y = float(face_parts[6])
            r_eye_y = float(face_parts[7])
            nose_y = float(face_parts[8])
            l_mouth_y = float(face_parts[9])
            r_mouth_y = float(face_parts[10])

            # the remaining values in the row contain these information #gender #smile #wearing glasses #head pose
            # the values start from one, we remove one to make them start from zero
            gender = int(face_parts[11]) - 1
            smile = int(face_parts[12]) - 1
            glasses = int(face_parts[13]) - 1
            pose = int(face_parts[14]) - 1

            ##############################################
            # face detection for a bigger rectangle area #
            ##############################################
            if extended:
                dist_ratio = 2.5
            else:
                dist_ratio = 1.25
            rect = detect_face_by_kpt(l_eye_x, l_eye_y, r_eye_x, r_eye_y, l_mouth_x, l_mouth_y, r_mouth_x, r_mouth_y, img_width=width, img_height=height,
                                      dist_ratio=dist_ratio)
            x_start, y_start, width, height = rect
            x_end = x_start + width
            y_end = y_start + height
            rectangle = rect

            ###################
            # sanity checking #
            ###################
            """
            Note: the sanity section (if set to true) writes to an output file the detected rectangle and the keypoint locations on them
                  to verify the keypoint locations and the rectangle are correct.
            """
            if sanity:
                source_dir = parent_dir + '/Preprocessings/face_detection/keypoint_face_detect' + '/kpt_detection_raw'
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
                #import pdb; pdb.set_trace()
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
            Y_name.append(file_jpg)
            Y_kpt_orig.append([l_eye_x, l_eye_y, r_eye_x, r_eye_y, nose_x, nose_y, l_mouth_x, l_mouth_y, r_mouth_x, r_mouth_y])
            Y_kpt.append([l_eye_x_norm, l_eye_y_norm, r_eye_x_norm, r_eye_y_norm, nose_x_norm, nose_y_norm, l_mouth_x_norm, l_mouth_y_norm, r_mouth_x_norm, r_mouth_y_norm])
            Y_rect.append(rectangle)
            Y_gl.append(glasses)
            Y_gen.append(gender)
            Y_sm.append(smile)
            Y_pose.append(pose)

            print "done with file %s" %file_path

    X_train = np.array(X_train)
    Y_train_name = np.array(Y_train_name)
    Y_train_kpt_orig = np.array(Y_train_kpt_orig)
    Y_train_kpt = np.array(Y_train_kpt)
    Y_train_rect = np.array(Y_train_rect)
    Y_train_gl = np.array(Y_train_gl)
    Y_train_gen = np.array(Y_train_gen)
    Y_train_sm = np.array(Y_train_sm)
    Y_train_pose = np.array(Y_train_pose)

    X_test = np.array(X_test)
    Y_test_name = np.array(Y_test_name)
    Y_test_kpt_orig = np.array(Y_test_kpt_orig)
    Y_test_kpt = np.array(Y_test_kpt)
    Y_test_rect = np.array(Y_test_rect)
    Y_test_gl = np.array(Y_test_gl)
    Y_test_gen = np.array(Y_test_gen)
    Y_test_sm = np.array(Y_test_sm)
    Y_test_pose = np.array(Y_test_pose)

    Y_train = OrderedDict()
    Y_train['name'] = Y_train_name
    Y_train['kpt_orig'] = np.float32(Y_train_kpt_orig)
    Y_train['kpt_norm'] = np.float32(Y_train_kpt)
    Y_train['face_rect'] = np.int16(Y_train_rect)
    Y_train['glasses'] = np.uint8(Y_train_gl)
    Y_train['gender'] = np.uint8(Y_train_gen)
    Y_train['smile'] = np.uint8(Y_train_sm)
    Y_train['pose'] = np.uint8(Y_train_pose)

    train_set = OrderedDict()
    train_set['X'] = X_train
    train_set['Y'] = Y_train

    Y_test = OrderedDict()
    Y_test['name'] = Y_test_name
    Y_test['kpt_orig'] = np.float32(Y_test_kpt_orig)
    Y_test['kpt_norm'] = np.float32(Y_test_kpt)
    Y_test['face_rect'] = np.int16(Y_test_rect)
    Y_test['glasses'] = np.uint8(Y_test_gl)
    Y_test['gender'] = np.uint8(Y_test_gen)
    Y_test['smile'] = np.uint8(Y_test_sm)
    Y_test['pose'] = np.uint8(Y_test_pose)

    test_set = OrderedDict()
    test_set['X'] = X_test
    test_set['Y'] = Y_test

    f_test.close()
    f_train.close()


    suffix = ''
    if extended:
        suffix = '_extended'

    train_name = 'MTFL_train_160by160'
    train_pkl_path = "%s/%s%s.pickle" %(out_dir, train_name, suffix)
    with open(train_pkl_path, 'wb') as fp:
        pickle.dump(train_set, fp)

    test_name = 'MTFL_test_160by160'
    test_pkl_path = "%s/%s%s.pickle" %(out_dir, test_name, suffix)
    with open(test_pkl_path, 'wb') as fp:
        pickle.dump(test_set, fp)
