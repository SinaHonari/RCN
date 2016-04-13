"""
This module converts the datasets float64 to float32 type
"""
import cv2
import numpy as np
import cPickle as pickle
from collections import OrderedDict
import random
import copy

class EOF(object):
    pass

def contrast_normalize(data_set):
    """
    This method performs the global contrast normalization on the data_set

    Parameters:
    -------------
        data_set: a 3D tensor of size (#sampels, #rows, #cols)
                  or
                  a 4D tensor of size (#sampels, #rows, #cols, #channels)

    Note: In both cases contrast normalization across the whole image is taken.
    """
    data_3D = False
    if len(data_set.shape)==3:
        n,r,c = data_set.shape
        # reshaping data_set to (#sampels, #rows * #cols)
        data_set = data_set.reshape(n, r * c)
        data_3D = True
    elif len(data_set.shape)==4:
        n,r,c,ch = data_set.shape
        # reshaping data_set to (#sampels, #rows * #cols * #channels)
        data_set = data_set.reshape(n, r * c * ch)
    else:
        msg = "input datast must be a 3D or 4D tensor!, received %iD input" %(len(data_set.shape))
        raise Exception(msg)

    # data_set is now in 2D (#samples, #pixels*channels)
    # getting the mean of the columns for each sample
    means = np.mean(data_set, axis=1)
    means = means.reshape(n, 1)
    data_set_m0 = data_set - means
    stds = np.std(data_set_m0, axis=1)
    stds = stds.reshape(n, 1)
    data_set_cn = data_set_m0 / (stds + 0.0000001)

    # assertions on the derived data
    means = np.mean(data_set_cn, axis=1)
    #assert np.all(abs(means)<1e-2)
    stds = np.std(data_set_cn, axis=1)
    #assert np.all(abs(stds - 1.0)<1e-2)

    if data_3D:
        data_set = data_set_cn.reshape(n, r, c)
    else:
        data_set = data_set_cn.reshape(n, r, c, ch)

    return np.float32(data_set)

def getSeed():
    # the seed for shuffling the data. This is done before the start of the program.
    SEED = 7345
    return SEED

def shuffleData(set_x, set_y):
        """
        this method shuffles the dataset

        Parameters:
        -------------
                set_x: is expected to be a tensor or matrix, which will
                       be shuffled by its rows (across samples)
                set_y: an OrderedDict.
                    For each element in the OrderedDict,
                    it the values of the key are shuffled
                    with the same order as the ones in the set_x
        """
        num_samples = len(set_x)

        # in order to always have the same split for the train and valid sets
        # a seed value is used, which always returns the same spllit
        SEED = getSeed()
        np.random.seed(SEED)
        indices = np.random.permutation(num_samples)

        # getting the shuffled sets
        shuffled_x = set_x[indices]

        shuffled_y = OrderedDict()
        if set_y:
            for key in set_y.keys():
                assert len(set_y[key]) == num_samples
                shuffled_y[key] = set_y[key][indices]

        return [shuffled_x, shuffled_y]

def splitData(set_x, set_y, split_size=1000):
        """
        this method gets set_x and set_y and splits them into two sets

        Parameters:
        -------------
                set_x: is expected to be a tensor or matrix, which will
                       be split by its rows (across samples)
                set_y: an OrderedDict.
                    For each element in the OrderedDict,
                    it the values of the key are split
                    with the same split_size as the ones in the set_x

                split_size: the size of the second set to be created
        """

        num_samples = len(set_x)
        assert num_samples > split_size

        set_x_1 = set_x[:-split_size]
        set_x_2 = set_x[-split_size:]

        set_y_1 = OrderedDict()
        set_y_2 = OrderedDict()
        if set_y:
            for key in set_y.keys():
                assert len(set_y[key]) == num_samples
                set_y_1[key] = set_y[key][:-split_size]
                set_y_2[key] = set_y[key][-split_size:]

        return [set_x_1, set_y_1, set_x_2, set_y_2]



def mergeData(dset1, dset2):
        """
        this method gets dset1 and dset2 and merges them together

        Parameters:
        -------------
                dset1: is a tuple of set_x and set_y
                dset2: same as dset1
        """
        set1_x, set1_y = dset1
        set2_x, set2_y = dset2

        set_x = np.concatenate((set1_x, set2_x), axis=0)
        set_y = OrderedDict()
        # considering only the keys that are common in both sets
        keys = [key for key in set1_y.keys() if key in set2_y.keys()]
        for key in keys:
            vals = []
            vals.extend(set1_y[key])
            vals.extend(set2_y[key])
            set_y[key] = np.array(vals)

        return [set_x, set_y]


def BGR2Gray(set_x):
    """
    This method takes input in cv2 BGR format and converts it to grayscale

    Parameters:
    -------------
            set_x: is expected to be a 4D tensor of shape (#sampels, #rows, #cols, 3)
                   or
                   an image (a 3D tensor) of shape  (#rows, #cols, 3)
                   Note: The input is expected to have 3 channels.

    return:
    --------
    unit8
        a 3D tensor or a matrix of gray-scale values.
        The output is of shape (#sampels, #rows, #cols) if the input is 4D
        and of shape (#rows, #cols) if the input is 3D
    """
    if len(set_x.shape)==3:
        # assertion on having 3 channels in the third dimension
        assert set_x.shape[2] == 3
        return cv2.cvtColor(set_x, cv2.COLOR_BGR2GRAY)

    elif len(set_x.shape)==4:
        # assertion on having 3 channels in the forth dimension
        assert set_x.shape[3] == 3
        set_x_gray = []
        for x in set_x:
            gray_img = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            r,c = gray_img.shape
            gray_img = gray_img.reshape(r,c,1)
            set_x_gray.append(gray_img)

        return np.array(set_x_gray)

    else:
        msg = "input datast must be a 3D or 4D tensor!, received %iD input" %(len(data_set.shape))
        raise Exception(msg)


def get_jittering(seed_rng, scale_size, translate_size, rotation_size, img_width):
    """
    This method returns uniform random values for scaling, translation, and rotation
    in the range given by user.

    Parameters:
    -------------
        scale_size: float in [0,1]
                    the maximum ratio by which image can be scaled.
                    a value of 0.2 return a random scale in [0.8, 1.2]
        translate_size:
                    the raletive maximum number of pixel by which image can be translated.
                    a value of 0.2 return a random translate size
                    [ -0.2 * img_width, 0.2 * img_width]
        rotation_size:
                    the maximum degree by which image can be translated.


    returns:
    ---------
        scale: float
               the random scale value in range [1 - scale_size, 1 + scale_size]
        transalte: float
               the random translate pixel size in range
               [- translate_size * img_width, translate_size * img_width]
        rotate: int
                the random rotation degree in the range [-rotation_size, rotation_size]
    """
    # getting random scaling size
    scale_jitter = seed_rng.uniform(-scale_size, scale_size)
    scale = 1.0 + scale_jitter
    # getting random translation size
    translate_jitter = seed_rng.uniform(-translate_size, translate_size, size=(2))
    translate = img_width * translate_jitter
    # getting random rotation size
    rotate = seed_rng.random_integers(-rotation_size, rotation_size)
    return [scale, translate, rotate]


def get_bbox(x_start, y_start, x_end, y_end):
    """
    This method returns the bounding box of a face.

    Parameters:
    -------------
        x_start: the x value of top-left corner of bounding box
        y_start: the y value of top-left corner of bounding box
        width : the x value of bottom-right corner of bounding box
        height: the y value of bottom-right corner of bounding box

    returns:
    --------------
    [x1, y1, x2, y2, x3, y3, x4, y4]
    the list of x and y values starting from the top-left corner and going clock, or counter-clock wise
    """
    x1 = x_start
    y1 = y_start
    x2 = x_end
    y2 = y_start
    x3 = x_end
    y3 = y_end
    x4 = x_start
    y4 = y_end

    return [x1, y1, x2, y2, x3, y3, x4, y4]

def get_src_pts_affine_transform(bbox):
    """
    This method gets a bounding box in the format [x1, y1, x2, y2, x3, y3, x4, y4]
    and returns a triangle representing the source points in the affine transformation

    returns:
    -------
    output is in the format array([[x1, y1], [x2, y2], [x3, y3]])
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox
    src_pts = np.float32([[x1, y1], [x2, y2], [x3, y3]])
    return src_pts

def get_dst_pts_affine_transform(width, height):
    """
    This method gets the width and height of the target images
    and returns a triangle representing the destination points in the affine transformation

    returns:
    -------
    output is in the format array([[x1, y1], [x2, y2], [x3, y3]])
    """

    dest_pts = np.float32([[0., 0.], [width, 0.], [width, height]])
    return dest_pts

def get_src_pts_perspective_transform(bbox):
    """
    This method gets a bounding box in the format [x1, y1, x2, y2, x3, y3, x4, y4]
    and returns a quadrilateral representing the source points in the perspective transformation

    returns:
    -------
    output is in the format array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox
    src_pts = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    return src_pts

def get_dst_pts_perspective_transform(width, height):
    """
    This method gets the width and height of the target images
    and returns a quadrilateral representing the destination points
    in the perspective transformation

    returns:
    -------
    output is in the format array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    """

    dest_pts = np.float32([[0., 0.], [width, 0.], [width, height], [0, height]])
    return dest_pts

def transform_Kpts(key_points, mapMatrix):
    """
    This methods get matrix of keypoints and returns a matrix of their affine transformed location

    Parameters:
    -------------
        key_points:
                  a matrix of key_point locations in the format (#key-points, 2)
        MapMatrix:
                  affine transformation of shape (2 * 3)

    returns:
    --------------
        A matrix of affine transformed key_point location in the format (#key-points, 2)
    """

    r,c = key_points.shape
    kpts = key_points.T
    # kpts is now in the format (2, #key-points)
    ones = np.ones(r)
    # adding a row of one to the kpts
    # kpts_vals is in shape (3, #key_points)
    kpts = np.vstack((kpts, ones))

    #new_kpts is in shape (2, #key_points)
    new_kpts = np.dot(mapMatrix, kpts)
    return new_kpts.T

def get_bbox_limits(bbox):
    """
    This module gets a bbox and transform it in this format [x_start, y_start, width, height]
    and transform it into this format [x_start, y_start, x_end, y_end]
    """
    bbox[:,2] += bbox[:,0]
    bbox[:,3] += bbox[:,1]
    return bbox


def block_image(seed_rng, images, target_width, target_height, fixed_block=False):
    """ Putting a black patch on image

    This module gets a tensor with the first dimension indicating the number
    of images and blocks randomly part of each image
    """
    num_img = images.shape[0]
    # defining min and max size of the block window
    min_size = 20
    max_size = 50
    if target_width == target_height == 40:
        min_size /= 2
        max_size /= 2
    for i in xrange(num_img):
        if fixed_block:
            rnd_row = max_size
            rnd_col = max_size
        else:
            # creating 2 random values in the range [min_size, max_size]
            block_rnd = seed_rng.random_integers(min_size, max_size, size=(2))
            rnd_row = block_rnd[0]
            rnd_col = block_rnd[1]
        # creating random initial (x,y) locations for the block
        rnd_start_row = seed_rng.random_integers(0, target_width - rnd_row)
        rnd_start_col = seed_rng.random_integers(0, target_height - rnd_col)
        images[i][rnd_start_row:rnd_start_row+rnd_row, rnd_start_col:rnd_start_col+rnd_col, :] = 0
    return images


def padRatio_to_pixels(pad_ratio, img_size):
    """
    This module changes the pad ratio
    to pixel location in the image

    parameters:
    -----------
        pad_ratio: a matrix where each value indicates
                   the ratio of the pad regio to the
                   total image size. Each row of pad_ratio
                   corresponds to an example (image)
                   and contains 4 values,
                   indicating the pad ratio in 4 sides of the image.
                   They are in this order:
                   [left, top, right, bottom]
        type pad_ratio: a matrix of float

        img_size: size of image (#rows or #columns)
                  Note: if img_size is one, the padRatio
                  should remain in the range of [0,1]
                  and only the last two values are changed
                  to reflect the place where the pad starts.
        type img_size: int

    returns:
    --------
        a list of 4 values indcating the last pixel location
        before the pad region starts. The indices start from zero
        so for an image of size 80 the range is [0, 79]

    """
    if img_size != 1:
        pad_pixels_float = pad_ratio * img_size
        pad_pixels = np.round(pad_pixels_float)
    else:
        pad_pixels = pad_ratio

    # reducing the last two elements from img_size to
    # get the pixel where the pad region starts
    # the border pixel is the image size minues the
    # number of padding pixels
    pad_pixels[:,-2:] = img_size - pad_pixels[:,-2:] # - 1
    return pad_pixels

def get_padded_region(set_y, img_size, rects):
    """
    This module gets the padded region after the bounding box detection in preprocess_once
    and returns the new bound_mask and pad_ratio.

    parameters:
    -----------
        set_y: an OrderedDict containing at least 'pad_ratio' and 'bound_mask'.
        type set_y: OrderedDict

        img_size: the lenght or width of the images
        type img_size: int

        rects: set of the detected rectangled (by preprocess_once) for all images.
        type rects: list
    """
    pad_ratios = set_y['pad_ratio']
    bound_masks = set_y['bound_mask']
    new_pad_ratios = []
    new_bound_masks = []
    pad_ratio_default = np.zeros(pad_ratios.shape[1])
    bound_mask_default = np.zeros(bound_masks.shape[1])
    # changing the pad_ratio to pad_pixels
    pad_pixels = padRatio_to_pixels(pad_ratios, img_size)

    assert len(rects) == pad_ratios.shape[0]
    assert len(rects) == bound_masks.shape[0]

    for index, rect in enumerate(rects):
        pad_ratio = pad_ratios[index]
        bound_mask = bound_masks[index]
        pad_pixel = pad_pixels[index]
        # there is a padded region
        if np.any(pad_ratio) > 0: # only do the process if padding is used
            ##########################
            # changing the pad_ratio #
            ##########################
            x_start, y_start, width, height = rect
            assert width == height
            x_end = x_start + width
            y_end = y_start + height
            old_x_pad_start, old_y_pad_start, old_x_pad_end, old_y_pad_end = pad_ratio

            # checking if the detected rectangle covers the padded region
            x_pad_start = 0
            x_pad_end = 0
            y_pad_start = 0
            y_pad_end = 0
            if x_start < pad_pixel[0]:
                x_pad_start = pad_pixel[0] - x_start
            if y_start < pad_pixel[1]:
                y_pad_start = pad_pixel[1] - y_start
            if x_end > pad_pixel[2]:
                x_pad_end = x_end - pad_pixel[2]
            if y_end > pad_pixel[3]:
                y_pad_end = y_end - pad_pixel[3]

            # normalizing the new pad
            new_pad = np.array([x_pad_start, y_pad_start, x_pad_end, y_pad_end])
            new_pad_ratio = new_pad/float(width)
            new_pad_ratios.append(new_pad_ratio)

            #########################################
            # changing the bound_mask for keypoints #
            #########################################
            if np.any(bound_mask) > 0:
                new_bound_mask = copy.copy(bound_mask)
                # checking if any of the previously padded regions is not padded anymore
                if old_x_pad_start > 0 and x_pad_start == 0:
                   ind = np.where(bound_mask == 1)
                   new_bound_mask[ind] = 0
                if old_y_pad_start > 0 and y_pad_start == 0:
                   ind = np.where(bound_mask == 2)
                   new_bound_mask[ind] = 0
                if old_x_pad_end > 0 and x_pad_end == 0:
                   ind = np.where(bound_mask == 3)
                   new_bound_mask[ind] = 0
                if old_y_pad_end > 0 and y_pad_end == 0:
                   ind = np.where(bound_mask == 4)
                   new_bound_mask[ind] = 0
                new_bound_masks.append(new_bound_mask)
            # there is no masked keypoint
            else:
                new_bound_masks.append(bound_mask_default)
        # there is no padded region
        else:
            new_pad_ratios.append(pad_ratio_default)
            new_bound_masks.append(bound_mask_default)

    return new_bound_masks, new_pad_ratios


def limit_x(x, limit):
    """
    This module limits the values to the range of [0,limit]
    """
    x_gr_limit = x > limit
    x_le_limit = x_gr_limit * limit + (1 - x_gr_limit) * x
    x_gr_zero = x > 0.0
    x_norm = x_gr_zero * x_le_limit
    return x_norm

def discretise_y(norm_kpt, dim):
    """
    this method makes the y_kpt_norm discretized to be put to the softmax values

    :type norm_kpt: numpy float matrix of shape (#batch, #kpt*2)
    :param norm_kpt: kpt values in the range of [0,1]

    :type dim: int
    :param dim: the dimentionality of the target picture

    returns: a numpy int matrix of shape (#batch, #kpt)
             with values in the range of [0, dim**2)
    """
    # make sure the values fall in the range [0,1]
    y_norm = limit_x(norm_kpt, 0.99999)   # Don't allow exactly 1

    x_pos = (y_norm[:,::2] * dim).astype(int)
    y_pos = (y_norm[:,1::2] * dim).astype(int)
    discrt_pos = y_pos * dim + x_pos
    return discrt_pos

def get_bound_mask(y_bound_mask):
    """
    this method gets a bound_mask matrix
    and returns the masks for each keypoint

    :type y_bound_mask: int
    : param y_bound_mask: a matrix of shape (#batch, #kpts)
                  where for each keypoint one
                  values is given:
                  0: no_boundary, 1: left boundary,
                  2: top boundary, 3: right boundary,
                  4: bottom boundary

    returns: a vector of dim (#batch * #kpts)
            where for each kpt a mask of 1 indicates the
            kpt is not outside of the border (usable in training)
            and a mask of 0 indicates the value is outside of the
            border (not usable in training)
    """
    #bound_mask is of shape (#batch, #kpts)
    zero_one_bound_mask = np.ones_like(y_bound_mask)
    zero_index = np.where(y_bound_mask > 0)
    zero_one_bound_mask[zero_index] = 0
    return zero_one_bound_mask

def mask_padded_kpts(kpts_norm, y_mask_border):
    """
    This method gets the kpts_norm and if they are not in the range of [0,1],
    it sets their corresponding element in mask_border
    to zero, indicating that the kpts is masked in the training phase.

    parameters:
    -------------
        1) kpt_norm: the normalized kpts values that are expected to be in the range [0, 1]
           kpt_norm is of shape (#batch, #kpts * 2 )
        2) y_mask_border: a matrix of shape (#batch, #kpts). Each value is either 1: indicating
                                the kpt is not padded or 0: indicating the kpt is in the pad and therefore masked.

    returns: a new y_mask_border that also masks the points that are not in the range [0,1]
    """
    # specifying which values are in the range [0,1]
    # vals_in_range is of shape (#batch, #kpts * 2)
    vals_in_range = (kpts_norm >= 0.) * (kpts_norm < 1.0)

    num_kpts = kpts_norm.shape[1]/2

    # vals_in_range_3D is of shape (#batch, #kpts, 2)
    vals_in_range_3D = vals_in_range.reshape(vals_in_range.shape[0], num_kpts, 2)

    # specifying which kpts have both x and y in the accepted range
    # kpts_in_range is a vector of dim (#batch, #kpts)
    kpts_in_range = vals_in_range_3D[:, :, 0] * vals_in_range_3D[:, :, 1]

    new_mask_border = kpts_in_range * y_mask_border

    return new_mask_border


def get_one_hot_map(kpts, dim):
    """
    This method gets the kpts and returns
    one-hot matrix with only one non-zero
    location.

    parameters:
        kpts: a matrix of shape (#batch, #kpts)
              with each value in the range of
              [0, dim**2)
        dim: the dimension (row or col) of the target feature_maps

    returns:
        3D tensor of shape (#batch, #kpts, #dim * #dim)
        of one-hot vectors where each vector of size
        dim*dim contains one non-zero value.
    """
    num_batch, num_kpts = kpts.shape

    # building the dimension zero of indices
    # ind_0 contains values in the range [0, num_batch)
    # each one repeated num_kpts times
    # [0, ..., 0, 1, ..., 1, num_batch-1, ..., num_batch-1]
    ind_0 = np.arange(num_batch)
    ind_0 = np.repeat(ind_0, num_kpts)

    # building the dimension one of indices
    # ind_1 contains values in the range [0, num_kpts)
    # which is tiled num_batch times
    # [0, 1, 2, ..., num_kpts-1, 0, 1, ..., nkpt-1, ..., 0, 1, ..., nkpt-1]
    ind_1 = np.arange(num_kpts)
    ind_1 = np.tile(ind_1, num_batch)

    # building the dimension one of indices
    ind_2 = kpts.flatten()

    one_hot_maps = np.zeros((num_batch, num_kpts, dim * dim))
    one_hot_maps[ind_0, ind_1, ind_2] = 1

    return one_hot_maps


def softmax_2D(x, use_float64=False):
    """Compute softmax values for each row in x.

    Parameters:
    -----------
    x: a 2D matrix
       each row represents a set of entries over which a softmax
       should be applied

    Returns:
    -------
    normalized softmax entries, where each row is normalized using softmax
    """
    row, col = x.shape
    max_per_row = np.max(x, axis=1).reshape(row, 1)
    if use_float64:
        max_per_row = max_per_row.astype('float64')
    e_x = np.exp(x - max_per_row)
    return e_x / e_x.sum(axis=1).reshape(row, 1)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_rnd_kpts_per_samples(num_batch, num_kpts, nMaps_shuffled):
    """
    This method gets num_batch, num_kpts, nMaps_shuffled
    and for each example in num_batch, selects as many as
    'nMaps_shuffled' random UNIQUE samples in the range [0, num_kpts)
    and returns them
    """
    # building the dimension zero of indices
    # ind_0 contains values in the range [0, num_batch)
    # each one repeated nMaps_shuffled times
    # [0, ..., 0, 1, ..., 1, num_batch-1, ..., num_batch-1]
    # ind_0 indicates which batch example should be shuffled
    ind_0 = np.arange(num_batch)
    ind_0 = np.repeat(ind_0, nMaps_shuffled)

    # building the dimension one of indices
    # ind_1 indicates which keypoints should be replaced
    ind_1 = []
    for ii in xrange(num_batch):
        # create 'nMaps_shuffled' unique random values in the range [0, num_kpt)
        randoms = random.sample(range(num_kpts), nMaps_shuffled)
        ind_1.append(randoms)
    ind_1 = (np.array(ind_1)).flatten()
    return ind_0, ind_1


def get_and_noise_one_hot_maps(kpts, pre_softmax_maps, dim, nMaps_shuffled,
                               rng, dropout_kpts, temperature):
    """
    This method gets the one-hot vectors for the true-kpt locations
    selects random kpts in each example and replaces the true position
    with a different random one-hot representation.

    parameters:
    -----------
        kpts: a matrix of shape (#batch, #kpts)
              with each value in the range of
              [0, dim**2)

        pre_softmax_maps: a matrix of shape (#batch_size, #kpts, #rows, #cols)
                          corresponding to the pre_softmax values of the RCN
                          model with each element representing the likelihood
                          of that element being the keypoint.

        dim: the dimension (row or col) of the target feature_maps

        nMaps_shuffled: the number of kpts that are jitteren in each example

        rng: numpy random state for drawing seeded random values

        dropout_kpts: a boolean indicating whether to completely mask the random
                   selected keypoints or not. if true, the maps corresponding to
                   those keypoints are all zeros.

        temperature: float scalar
            a value in the range [0,1] indicating the temperature that is assigned
            to the softmax distribution of each keypoint.

    returns:
    --------
        one_hot_maps_4D: 4D tensor of shape (#batch, #kpts, #dim, #dim)
                         of one-hot (#dim, #dim) matrices where for each
                         batch example a random one-hot matrix replaces the
                         true one-hot matrix.
        mask: a 2D matrix of shape (#batch, #kpts) having ones for the randomly
              jittered kpts and zero in other locations.
    """
    # getting one_hot maps of true locations
    # one_hot_maps is of shape (#batch, #kpts, #dim * #dim)
    one_hot_maps = get_one_hot_map(kpts, dim)

    num_batch, num_kpts = kpts.shape

    ##########################################
    # generating random locations for kpts   #
    # This corresponds to the kpts that will #
    #            be jittered                 #
    ##########################################
    ind_0, ind_1 = get_rnd_kpts_per_samples(num_batch, num_kpts, nMaps_shuffled)

    # setting selected kpt indices in one_hot_maps to zero
    one_hot_maps[ind_0, ind_1, :] = 0

    if not dropout_kpts:
        if temperature == 0:
            # In this case a uniform sample over the entire map is taken
            # building the dimension two of indices
            # getting random int number in the range [0, dim**2)
            # for random keypoint position genration
            ind_2 = np.random.randint(low=0, high=dim*dim, size=nMaps_shuffled * num_batch)
        else:
            bch, ch, row, col = pre_softmax_maps.shape
            pre_softmax_maps_2D = pre_softmax_maps.reshape(bch * ch, row * col)
            softmax_probs = softmax_2D(pre_softmax_maps_2D * temperature, use_float64=True)
            softmax_probs_3D = softmax_probs.reshape(bch, ch, row * col)
            ind_2 = []
            for smpl in xrange(nMaps_shuffled * num_batch):
                bch_index = ind_0[smpl]
                kpt_index = ind_1[smpl]
                probs = softmax_probs_3D[bch_index, kpt_index]
                sum_probs = sum(probs)
                if sum_probs != 1.0:
                    # in this case a small difference exists between sum_probs and 1.0
                    # which is added to the biggest element of the probs list.
                    # this process is repeated as long as the sum equals to 1.
                    max_index = np.argmax(probs)
                    probs[max_index] += 1.0 - sum_probs
                ind_2_smpl = np.random.choice(a=dim*dim, size=1, replace=True, p=probs)
                ind_2.extend(ind_2_smpl)

        # setting selected kpt and position indices in kpt_maps to one
        one_hot_maps[ind_0, ind_1, ind_2] = 1

    #################
    # getting masks #
    #################
    # the mask indicates which points are jittered
    # 1 for jittered (noised) points and 0 otherwise
    mask = np.zeros((num_batch, num_kpts))
    mask[ind_0, ind_1] = 1

    one_hot_maps_4D = one_hot_maps.reshape(num_batch, num_kpts, dim, dim)
    return one_hot_maps_4D, mask
