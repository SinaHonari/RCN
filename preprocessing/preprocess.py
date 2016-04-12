"""
This module does the preprocessing on faces
"""
import os
import cv2
import numpy as np
from PIL import Image
import cPickle as pickle
from collections import OrderedDict
from RCN.preprocessing.affine_transform import transform_bbox
from RCN.utils.polar_cartesian_convert import pol_to_cart, cart_to_pol
from RCN.preprocessing.tools import (shuffleData, splitData,
                                            contrast_normalize, BGR2Gray,
                                            get_jittering, get_bbox,
                                            get_src_pts_affine_transform,
                                            get_dst_pts_affine_transform,
                                            transform_Kpts, get_bbox_limits,
                                            block_image, get_padded_region)
from RCN.preprocessing.detect_face import detect_dataset_faces


def preprocess_once(set_x, set_y, gray_scale=True, dist_ratio=3.8/4.0):
    """
    This method detects the face in the gray-scale (if user wants)
    and detects a bounding box around the face using the keypoints.
    It is done once for all the dataset, at the beginning of training.

    Parameters:
    -----------
            set_x: 4D numpy ndarray
               a 4D tensor of size (#sampels, #rows, #cols, #channels)
               channel is expected to be 3.
            set_y: OrderedDict
               an OrderedDict with at least the following component.
               Each component has as many rows as the number of samples.
                    'kpt_norm' : the normalized keypoint positions
                        such that each x or y is in the range of [0,1]

            gray_scale: bool
                        if true, the image is returned in gray-scale
            dist_ratio: float
                        the ratio by which to detect the face.
                        smaller values corresponds to a more closed-up views.

    returns:
    --------
        set_x: a 4D tensor of size (#sampels, #rows, #cols, #channels)
           #channels gets equal to 1, if the input is set to gray-scale=True

        set_y: OrderedDict
           It is equivalent to the input but
           replaces the value of the following key:
                'face_rect' : the new detected rectangle
                              around the face with values:
                                  rect_start_x, rect_start_y, rect_width
                                  (rect_end_x - rect_start_x),
                                  rect_height (rect_end_y - rect_start_y)
    """
    print "detecting faces"
    if gray_scale:
        print "images are grayscaled"
        set_x = BGR2Gray(set_x)

    # detecting the inner bonding box for face jittering #
    print "detecting the rectangle around the face"
    # third dimension correcponds to the number of rows or cols
    img_size = set_x.shape[2]
    rects = detect_dataset_faces(set_y, img_size, dist_ratio)

    # getting the padded region for 300-W dataset
    kpts_norm = set_y['kpt_norm']
    num_vals_300W = 136
    if kpts_norm.shape[1] == num_vals_300W:
        print "getting the new padded region"
        bound_mask, pad_ratio = get_padded_region(set_y, img_size, rects)
        set_y['bound_mask'] = np.int8(bound_mask)
        set_y['pad_ratio'] = np.array(pad_ratio)

    set_y['face_rect'] = np.int16(rects)

    return [set_x, set_y]


def flip_img(img):
    return np.fliplr(img)


def crop_img(img, x_start, y_start, x_end, y_end):
    # the method for cropping an image
    return img.crop((x_start, y_start, x_end, y_end))


def resize_img(img, size):
    # the methods for downsampling
    return img.resize(size, Image.ANTIALIAS)


def preprocess_iter(set_x, set_y, seed_rng, target_dim=(80, 80), jitter=True,
                    scale_mul=0.5, translate_mul=0.5, sanity=False,
                    use_lcn=False, dset='300W', block_img=False,
                    fixed_block=False, rotation=10, get_rotation=False,
                    rotation_set=None):
    """
    This method affine transforms the face bounding box.
    transform the image based on the affine transformation.
    affine transform the keypoints and gets their normalized locations
    it also gets the ocular distance.
    Finally it does the contrast normalization on the image.

    If the jitter is False, then the method
    just does downsampling for the images
    and gets the normalized keypoint positions and the ocular_distance.

    For the validation and test sets, it should be done once at the
    beginning of training, with jittering being set to False.
    for the train set, if the jittering is not needed,
    it should be done at the beginning of each epoch

    Parameters:
    -----------
        set_x: a 4D tensor of size (#sampels, #rows, #cols, #channels)
        set_y: OrderedDict
           an OrderedDict with at least the following component.
                Each component has as many rows as the number of samples.
                'kpt_norm' : the normalized keypoint positions such that
                    each x or y is in the range of [0,1]
                'face_rect' : the detected rectangle around the face
                     with values:
                    rect_start_x, rect_start_y, rect_width (rect_end_x -
                    rect_start_x), rect_height (rect_end_y - rect_start_y)
        target_dim: tuple of two ints
            a tuple with two ints indicating the width
            and height of the target image
        gray_scale: bool
            if true, the image is returned in gray-scale
        dist_ratio: float
            the ratio by which to detect the face.
            smaller values corresponds to a more closed-up views.
        sanity: bool
            a flag indicating whether this run is for a sanity check of
            the pre-processing or not. If set to True, it plots the transformed
            key-points in the new image and writes the results to jpg files.
            If you want to check this part, when calling preprocess_iter set
            sanity=True. in order to use this method with the sanity_check=True
            , just call check_preprocess method in this module.
        dset: str
            The dataset to be jittered

    returns:
    --------
        set_x_cn: a 4D tensor of size (#sampels, #rows, #cols, #channels)
                   basically the jittered or downsampled input.

        set_y: OrderedDict
           It is equivalent to the input but replaces the
                values of the following two keys:
                'kpt_norm'  : the normalized keypoint positions
                     of the jittered or downsampled face
                     such that each x or y is in the range of [0,1]
                'ocular_dist' : the inter_ocular distance between
                       the new normalized eye locations.
                       This value is also in the range of [0,1]
    """
    # if rotation degrees are provided by rotation_set,
    # get_rotation shoule be False
    if rotation_set is not None:
        assert get_rotation is False
    # if get_rotation is True (the rotation degree is asked for),
    # rotation_set should be None
    if get_rotation:
        assert rotation_set is None

    batch_size, rows, cols, channels = set_x.shape
    # assuming img is rectangular
    img_size = set_x.shape[2]
    rects = set_y['face_rect']
    batch_kpts = set_y['kpt_norm']

    set_x_jttr = []
    kpts_jttr = []
    ocular_dist_jttr = []

    l_eye_1_indx = 37 - 1
    l_eye_2_indx = 40 - 1
    r_eye_1_indx = 43 - 1
    r_eye_2_indx = 46 - 1

    ################
    # sanity check #
    ################
    if sanity:
        dir_detected = 'sanity_check'

        try:
            os.makedirs(dir_detected)
        except OSError:
            if not os.path.isdir(dir_detected):
                raise
    ####################
    # end sanity check #
    ####################

    ######################################
    # getting the rotation degree of img #
    ######################################
    # the keypoints are in this order l_eye_x, l_eye_y, r_eye_x, r_eye_y
    l_eye_x = batch_kpts[:, 0]
    l_eye_y = batch_kpts[:, 1]
    r_eye_x = batch_kpts[:, 2]
    r_eye_y = batch_kpts[:, 3]
    deltaY = r_eye_y - l_eye_y
    deltaX = r_eye_x - l_eye_x
    # deg_init for positive value is clock-wise and
    # for negative value is counter clock-wise
    deg_init = cart_to_pol(deltaX, deltaY)
    # deg_init = np.arctan2(deltaY, deltaX) * 180 / np.pi

    polar_list = []

    for i in xrange(batch_size):
        img_npy = set_x[i]
        rect = rects[i]
        x_start, y_start, width, height = rect
        x_end = x_start + width
        y_end = y_start + height
        # kpts is a row with size (#ketpoints * 2) , which should change to
        # a matrix of size (#ketpoints, 2)
        kpts = batch_kpts[i]

        # flip = False
        #######################
        # jittering the image #
        #######################
        if jitter:
            # choose with prob 0.5 to flip image or not
            # flip = seed_rng.binomial(n=1, p=0.5)
            scale_size = 0.2 * scale_mul  # maximum scaling ratio of the bbox
            # maximum translation ratio of bbox
            translate_size = 0.2 * translate_mul
            rotation_size = rotation  # maximum rotation degree
        else:  # in this case only downsampling is done.
                # This is useful for the valid or test set.
            scale_size = 0.0  # maximum scaling ratio of the bbox
            translate_size = 0.0  # maximum translation ratio of bbox
            rotation_size = 0  # maximum rotation degree

        # getting scale, translation and rotion random values
        # for jittering the bounding box
        scale, translate, rotate = get_jittering(seed_rng, scale_size,
                                                 translate_size, rotation_size,
                                                 width)

        ###############################################
        # checking if the rotation degree is provided #
        ###############################################
        if rotation_set is not None:
            # getting the negattion of the values passed by list
            scale = 1.0
            translate = [0.0, 0.0]
            rotate = -rotation_set[i]

        ############################################
        # getting the degree that image is rotated #
        ############################################
        # deg_target change for positive value is counter clock-wise
        # and for negative value is clock-wise
        # Note: pass the negated deg_target to transform_bbox method
        deg_target = rotate - deg_init[i]
        polar_list.append(deg_target)

        # getting bounding box in (x1,y1,x2,y2,x3,y3,x4,y4) format
        bbox = get_bbox(x_start, y_start, x_end, y_end)

        # affine transforming the bbox by scaling, transformation, and rotation
        # rot_bbox is the affine transformed bounding box
        rot_bbox = transform_bbox(bbox, angle=rotate, translation=translate,
                                  scalefactor=scale)

        if get_rotation:
            # rotating the images with the orignal size and
            # cropping them later
            affine_width, affine_height = rows, cols
        else:
            affine_width, affine_height = target_dim

        affine_dim = (affine_width, affine_height)


        # Affine_transform
        # getting the source and destination triangles
        # for affine transformation
        src_pts = get_src_pts_affine_transform(rot_bbox)
        dest_pts = get_dst_pts_affine_transform(affine_width,
                                                affine_height)

        # Applying the affine transformation matrix from src_pts to
        # dest_pts.
        # mapMatrix is a (2 * 3) matrix for affine transformation
        mapMatrix = cv2.getAffineTransform(src_pts, dest_pts)

        # Applygin the affine transformation to the image
        # a list of border types can be found here
        # http://docs.opencv.org/modules/imgproc/doc/filtering.html#int
        # borderInterpolate(int p, int len, int borderType)
        # img_jttr is the jittered image
        img_jttr = cv2.warpAffine(src=img_npy, M=mapMatrix,
                                  dsize=affine_dim,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)

        num_rows = img_jttr.shape[1]

        ################################################
        # applying the transformation to the keypoints #
        ################################################
        # the kpts are normalized in the range [0,1], they should be change to
        # values in the range of the image size
        kpts = kpts * img_size
        num_kpts = kpts.shape[0]/2
        # the locations for keypoints are in this format: [x1 y1 ... x5 y5] for
        # left eye, right eye, nose, left mouth corner, right mouth corner.
        # they should be changed to [[x1 y1] ... [x5 y5]]
        kpts = kpts.reshape(num_kpts, 2)

        # new_kpts is the transformed keypoints
        # in the shape (#key-points, 2)
        new_kpts_mat = transform_Kpts(kpts, mapMatrix)

        # reshaping new_kpts to a row with size (#ketpoints * 2)
        new_kpts = new_kpts_mat.reshape(num_kpts * 2)

        # normalizing new_kpts to be in the range [0,1]
        kpts_norm = new_kpts / affine_width
        kpts_jttr.append(kpts_norm)

        ###################
        # sanity checking #
        ###################
        if sanity:
            out_file_path = dir_detected + '/' + str(i) + '.jpg'
            for kpt in new_kpts_mat:
                c, r = int(kpt[1]), int(kpt[0])
                if c > 0 and c < num_rows and r > 0 and r < num_rows:
                    if img_jttr.shape[-1] == 3:
                        img_jttr[c, r] = (0, 255, 0)
                    else:  # it has a len of 2
                        img_jttr[c, r] = 1.  # filling with white value

            cv2.imwrite(out_file_path, img_jttr)
            if (i + 1) % 400 == 0:
                import pdb; pdb.set_trace()
        #######################
        # end sanity checking #
        #######################

        # getting the new ocular distance
        if dset == 'MTFL':
            l_eye_x_norm, l_eye_y_norm, r_eye_x_norm, r_eye_y_norm =\
                kpts_norm[:4]
        elif dset == '300W':
            kpts_3D = kpts_norm.reshape(num_kpts, 2)
            # getting left eye information
            l_eye1_x, l_eye1_y = kpts_3D[l_eye_1_indx]
            l_eye2_x, l_eye2_y = kpts_3D[l_eye_2_indx]
            l_eye_x_norm = np.mean((l_eye1_x, l_eye2_x))
            l_eye_y_norm = np.mean((l_eye1_y, l_eye2_y))

            # getting right eye information
            r_eye1_x, r_eye1_y = kpts_3D[r_eye_1_indx]
            r_eye2_x, r_eye2_y = kpts_3D[r_eye_2_indx]
            r_eye_x_norm = np.mean((r_eye1_x, r_eye2_x))
            r_eye_y_norm = np.mean((r_eye1_y, r_eye2_y))
        else:
            raise Exception('process_iter got neither MTFL nor 300W, \
                             unknown dataset %s' % (dset,))

        eye_distance_norm = np.sqrt((r_eye_x_norm - l_eye_x_norm)**2 +
                                    (r_eye_y_norm - l_eye_y_norm)**2)
        # assert eye_distance_norm <= 1.0
        ocular_dist_jttr.append(eye_distance_norm)
        set_x_jttr.append(img_jttr)

    # setting the information in the correct format
    set_x_jttr = np.array(set_x_jttr)

    # creating new set_y
    set_y_new = OrderedDict()
    for key in set_y.keys():
        set_y_new[key] = set_y[key]
    kpts_jttr = np.float32(kpts_jttr)
    set_y_new['kpt_norm'] = kpts_jttr
    set_y_new['ocular_dist'] = np.float32(ocular_dist_jttr)

    #############################################
    # detecting the face for the rotation model #
    # and cropping the image + downsampling it  #
    #############################################
    if get_rotation:
        set_x_cropped = []
        set_kpts = OrderedDict()
        set_kpts['kpt_norm'] = kpts_jttr
        rects = detect_dataset_faces(set_kpts, affine_width, 0.6)
        rects_limit = get_bbox_limits(rects)
        for i in xrange(set_x_jttr.shape[0]):
            img_cv = set_x_jttr[i]
            xs, ys, xe, ye = rects_limit[i]
            #####################################
            # converting from cv2 to PIL format #
            #####################################
            # converting from BGR to RGB
            if len(img_cv.shape) == 3:
                img_RGB = img_cv[:, :, ::-1].copy()
            elif len(img_cv.shape) == 2:
                img_RGB = img_cv
            # making sure values are uint8
            img_RGB = np.uint8(img_RGB)
            img = Image.fromarray(img_RGB)

            # cropping the image
            cropped_img = crop_img(img, xs, ys, xe, ye)
            # downsampling image
            resized_img = resize_img(cropped_img, size=target_dim)

            #####################################
            # converting from PIL to cv2 format #
            #####################################
            if channels == 3:
                # in order to convert PIL image to cv2 format,
                # the image should be converted to RGB first
                img_rgb = resized_img.convert('RGB')
                # getting the numpy array from image,
                # this numpy array is in RGB format
                img_cv2 = np.asarray(img_rgb)
                # cv2 images are in BGR fortmat,
                # so RGB should be changed to BGR
                img_crop = img_cv2[:, :, ::-1].copy()
            elif channels == 1:
                img_crop = np.asarray(resized_img)
            set_x_cropped.append(img_crop)
        set_x_jttr = np.array(set_x_cropped)

    # if img_npy is gray-scaled, then warpAffien transforms the img_npy of
    # shape (#rows, #cols, 1) to img_jttr of shape (#rows, #cols)
    # in order to add the channel, the data should be put back to 3D
    target_width, target_height = target_dim
    set_x_jttr = set_x_jttr.reshape(set_x_jttr.shape[0], target_width,
                                    target_height, channels)
    set_x_jttr_orig = set_x_jttr.copy()

    ######################
    # blocking the image #
    ######################
    if block_img:
        set_x_jttr = block_image(seed_rng=seed_rng, images=set_x_jttr,
                                 target_width=target_width,
                                 target_height=target_height,
                                 fixed_block=fixed_block)

    save_img = False
    if save_img:
        for i in xrange(100):
            name = 'img_%s.jpg' % (i,)
            cv2.imwrite(name, set_x_jttr[i])
    #import pdb; pdb.set_trace()

    # building the orderedDict for the rotation case
    set_y_rotate = OrderedDict()
    set_y_rotate['pol'] = np.float32(polar_list)
    cart_list = pol_to_cart(polar_list)
    set_y_rotate['cart'] = np.transpose(np.float32(cart_list))

    ####################################
    # do global contrast normalization #
    ####################################
    if not use_lcn:
        # sys.stderr.write("using local contrast normalization\n")
        set_x_cn = contrast_normalize(set_x_jttr)
    else:
        set_x_cn = set_x_jttr

    # if get_rotatioe is True, set_y_rotate
    # is returned which contains the polar and
    # cartesian information of the rotated images
    # in set_x_cn
    if get_rotation:
        return [set_x_cn, set_y_rotate]
    else:
        return [set_x_cn, set_y_new]
