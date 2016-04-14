# This module takes the weights of a convnet and
# sets the convnet weights to them, then it gives the convnet
# some sample images and plots the
# model's predicted keypoints on those images.
#
# Note: This module draws the keypoints on the downsampled 80*80 rgb images

import numpy as np
from RCN.models.create_procs import get_data
from RCN.preprocessing.tools import shuffleData, splitData
from RCN.preprocessing.preprocess import preprocess_iter, preprocess_once
from RCN.preprocessing.tools import EOF, padRatio_to_pixels
from RCN.utils.convnet_tools import (create_TCDCN_obejct, get_error_kpt_avg,
                                     get_one_hot_predictions)
import RCN
from collections import OrderedDict
import argparse
import cv2
import os
from PIL import Image
import cPickle as pickle
face_path = os.path.dirname(RCN.__file__)
from copy import deepcopy

def resize_img(img, size):
   # the methods for downsampling
   return img.resize(size, Image.ANTIALIAS)

def tile_images(sample_num, draw_set, out_path, row_size, col_size):
    if sample_num == 6:
        rows = 1
        cols = 6
    else:
        rows = 4
        cols = 5

    gap_sz = 5
    gap_cols = (cols - 1) * gap_sz
    gap_rows = (rows - 1) * gap_sz
    total_img = sample_num / float(rows * cols)
    total_img = int(np.ceil(total_img))
    index = 0

    for kk in xrange(total_img):
        # creating a white image
        new_im = Image.new('RGB', (cols*col_size + gap_cols, rows*row_size + gap_rows), "white")
        for i in xrange(0, rows * row_size + gap_rows, row_size + gap_sz):
            for jj in xrange(0, cols * col_size + gap_cols, col_size + gap_sz):
                index+=1
                if index > sample_num:
                    break
                name = "%s_%i_downsampled.png" %(draw_set, index)
                file_path = "%s/%s" %(out_path, name)
                img = Image.open(file_path)
                new_im.paste(img, (jj, i))

            # checking if index is passed in
            # i loop
            if index > sample_num:
                break
        target_path = "%s/tiled_%s_%i.png" %(out_path, draw_set, kk)
        new_im.save(target_path)

def get_subsets(set_x, set_y, sample_num):
    set_x = set_x[:sample_num]
    for key in set_y.keys():
        set_y[key] = set_y[key][:sample_num]
    return [set_x, set_y]

def get_indices(set_x, set_y, indices):
    set_x = set_x[indices]
    for key in set_y.keys():
        set_y[key] = set_y[key][indices]
    return [set_x, set_y]

def drawpoints(img, kpt_conv, kpt_true, plot_colored_kpt, magnify=False):
    #plotting the convNet predicted locations
    kpt_true = map(int, kpt_true)
    kpt_true = np.array(kpt_true)
    kpt_conv = kpt_conv.reshape(kpt_conv.shape[0]/2 , 2)
    #plotting the true keypoint locations
    kpt_true =kpt_true.reshape(kpt_true.shape[0]/2 , 2)

    if magnify:
        mul = 4
        row, col = img.shape[:2]
        resized_image = cv2.resize(img, (row*mul, col*mul))
        img = resized_image
        kpt_true *= mul
        kpt_conv *= mul

    if not plot_colored_kpt:
        for kpt1, kpt2 in zip(kpt_true, kpt_conv):
            x1, y1 = kpt1[0], kpt1[1]
            x2, y2 = kpt2[0], kpt2[1]
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.line(img, pt1, pt2, color=(0,255,255))

        for kpt in kpt_true:
            # plotting the estimated keypoints in green
            if magnify:
                cv2.circle(img,(int(kpt[0]),int(kpt[1])), 3, (0,255,0), -1)
            else:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.rectangle(img, pt1=(x-1, y-1), pt2=(x+1, y+1), color=(0,255,0))

        for kpt in kpt_conv:
            # plotting the estimated keypoints in blue
            if magnify:
                cv2.circle(img,(int(kpt[0]),int(kpt[1])), 3, (0,0,255), -1)
            else:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.rectangle(img, pt1=(x-1, y-1), pt2=(x+1, y+1), color=(0,0,255))
    else:
        for i, kpt in enumerate(kpt_conv):
            if i==0:
                color = (0, 0, 255)
            elif i==1:
                color = (0, 255, 255)
            elif i==2:
                color = (0, 255, 0)
            elif i==3:
                color = (51, 153, 255)
            else:
                color = (255, 147, 23)
            x, y = int(kpt[0]), int(kpt[1])
            cv2.rectangle(img, pt1=(x-3, y-3), pt2=(x+3, y+3), color=color, thickness=-1)

    return img

def annotateKpts(img, kpt_conv, kpt_true):
    row, col = img.shape[:2]

    text_size = 110
    img_new = np.zeros((row, col + 2*text_size, 3))
    img_new[ :, text_size: text_size+col, :] = img
    font = cv2.FONT_HERSHEY_SIMPLEX
    #plotting the convNet predicted locations
    kpt_true = map(int, kpt_true)
    kpt_true = np.array(kpt_true)
    kpt_true = kpt_true.reshape(kpt_true.shape[0]/2 , 2)
    kpt_conv = kpt_conv.reshape(kpt_conv.shape[0]/2 , 2)
    for i, (k_true, k_conv) in enumerate(zip(kpt_true, kpt_conv)):
        # plotting the estimated keypoints in blue
        x_true = k_true[0]
        y_true = k_true[1]
        x_conv = k_conv[0]
        y_conv = k_conv[1]
        x_diff = np.abs(x_true - x_conv)
        y_diff = np.abs(y_true - y_conv)
        x_msg = "x%i: %i, %i, %i" %(i, x_true, x_conv, x_diff)
        y_msg = "y%i: %i, %i, %i" %(i, y_true, y_conv, y_diff)
        cv2.putText(img_new, x_msg, (3, 10*(i+1)), font, 0.4, (255,255,255))
        cv2.putText(img_new, y_msg, (text_size + row + 3, 10*(i+1)), font, 0.4, (255,255,255))

    return img_new

def get_pickle_indices(picke_path):
    with open(pickle_path, 'rb') as fp:
        vals = pickle.load(fp)
        pickle_indices = vals['samples_vec']
    return pickle_indices, vals

def save_error_results(array, set_name, sample_num):
    array_asc = np.sort(array)[::-1]
    array_asc_indx = np.argsort(array)[::-1]
    print "error values for set %s are %s" %(set_name, array_asc[:sample_num])
    print "error indices for set %s are %s" %(set_name, array_asc_indx[:sample_num])
    out_str_path = "%s/%s_sorted_error.txt" %(out_path, set_name)
    out_str=open(out_str_path,'w')
    out_str.write("error, index\n")
    for err, indx in zip(array_asc, array_asc_indx):
        out_str.write("%s, %s\n" %(err, indx))

    out_str_path = "%s/%s_sorted_indices.pickle" %(out_path, set_name)
    with open(out_str_path, 'wb') as fp:
        pickle.dump(array_asc_indx, fp)
    print "done with %s" %(set_name)
    return array_asc_indx[:sample_num]

def eval_test_set(tcdcn, params, set_x, set_y, set_name, sample_num, dataSet, cfNet_model):
    rng_seed  = np.random.RandomState(0)
    scale_mul = 0.0
    translate_mul = 0.0

    target_dim =  params['target_dim']
    td = (target_dim, target_dim)
    if dataSet == 'MTFL':
        dist_ratio = 3.8/4.0
    else:
        dist_ratio = params['dist_ratio']

    set_x2, set_y2 = preprocess_once(set_x, set_y, dist_ratio=dist_ratio, gray_scale=params['gray_scale'])
    # using preprocess_iter to prepare set_x2 and set_y2 as the model expect it
    set_x2, set_y2 = preprocess_iter(set_x2, set_y2, rng_seed, jitter=False, scale_mul=scale_mul,
                                     translate_mul=translate_mul, target_dim=td, sanity=False, dset=dataSet,
                                     use_lcn=params['use_lcn'])

    error_kpt_avg_all, error_kpt_avg_mean = get_error_kpt_avg(tcdcn=tcdcn, params=params, test_set_x=set_x2,
                                                              test_set_y=set_y2, dataSet=dataSet,
                                                              set_name=set_name, batch_size=1, cfNet_model=cfNet_model)
    array_asc_indx = save_error_results(error_kpt_avg_all, set_name, sample_num)
    return array_asc_indx, error_kpt_avg_all


def valid_remaining_dim(shape_1, shape_2):
    """
    This method checks that the dimension 2 to the end of
    shape of are the same.
    """
    if len(shape_1) != len(shape_2):
        return False
    for dim in np.arange(1, len(shape_1)):
        if shape_1[dim] != shape_2[dim]:
            return False
    return True

def append_orderedDict(list_of_dict):
    """
    This method receive a list of orderedDict
    and returns a merge of them. It only considers
    keys that exist in all of them.
    """
    all_sets = OrderedDict()
    for key in list_of_dict[0].keys():
        all_sets[key] = list_of_dict[0][key]
        for mylist in list_of_dict[1:]:
            if key not in mylist.keys():
                del all_sets[key]
                break
            elif not valid_remaining_dim(all_sets[key].shape, mylist[key].shape):
                del all_sets[key]
                break
            else:
                all_sets[key] = np.concatenate((all_sets[key], mylist[key]), axis=0)
    return all_sets

def draw_points_raw(out_path, annotate_kpts=True, high_res=True, max_errors=True, sample_num=20,
                    plot_colored_kpt=False, indices_given=False, cfNet_path="", mult_probs=False,
                    merge_sets=False, pickle_path=""):
    """
    This method draw points on the test set, when there are two steps for face-detection and
    downsampling. So, when the dataset is created, a face-detection is done and the data is downsampled to
    orig_downsample size (this is done before pickling the data). Then, later,
    at train-time, a second face detection is performed and the data is downsampled
    to second_downsample size. In both down_sampling stages, the kpt_norm keeps a value in the range [0,1]. So, the relative
    location is the same and just normalized in the downsampled case, compared to the detected face locations.

    The process of finding the original keypoint locations is as follows:
        for the 2nd face detection, multiply the normalized key-point locations in the bounding_box size
        and add to those values to location of the top-left position of the bounding_box. This gives the
        positions before the 2nd face detection. Then, normalize the
        positions by the downsampling (img) size of the first face-detection to get them normalized. Finally,
        multiply the normalized key-point locations in the bounding_box size of the first face detection
        and add to those values the location of the top-left position of that bounding_box.
    """

    # setting the conv params to the weights
    tcdcn, params = create_TCDCN_obejct(pkl_param_file)
    tcdcn.load_params(pkl_param_file)

    tcdcn_cfNet, params_cfNet = None, None
    if cfNet_path != "":
        print "loading params of cfNet"
        tcdcn_cfNet, params_cfNet = create_TCDCN_obejct(cfNet_path)
        tcdcn_cfNet.load_params(cfNet_path)
    cfNet_model = (tcdcn_cfNet, params_cfNet)

    if params['paral_conv'] in [2, 5, 6] or params['denoise_conv'] in [1, 2]:
        params['mask_MTFL'] = 0
        params['mask_300W'] = 1
        dataSet = '300W'
        num_kpt = 68
    elif params['paral_conv'] in [1, 3, 4]:
        params['mask_300W'] = 0
        params['mask_MTFL'] = 1
        dataSet = 'MTFL'
        num_kpt = 5

    ##############################################
    # getting the 1st bounding box test set data #
    ##############################################
    sets = get_data(**params)
    Train, Valid, Test = sets
    sets = OrderedDict()

    train_set_x, train_set_y = Train[dataSet]
    sets['train'] = OrderedDict()
    sets['train']['X'] = train_set_x
    sets['train']['Y'] = train_set_y
    sets['train']['indices'] = []
    sets['train']['name'] = '1_train'

    valid_set_x, valid_set_y = Valid[dataSet]
    sets['valid'] = OrderedDict()
    sets['valid']['X'] = valid_set_x
    sets['valid']['Y'] = valid_set_y
    sets['valid']['indices'] = []
    sets['valid']['name'] = '2_valid'

    for test_set in np.sort(Test[dataSet].keys()):
        test_set_x, test_set_y = Test[dataSet][test_set]
        if merge_sets:
            if not 'all_sets' in sets.keys():
                sets['all_sets'] = OrderedDict()
                sets['all_sets']['X'] = []
                sets['all_sets']['Y'] = []
                sets['all_sets']['name'] = 'all_sets'
            sets['all_sets']['X'].extend(test_set_x)
            sets['all_sets']['Y'].append(test_set_y)
        else:
            sets[test_set] = OrderedDict()
            sets[test_set]['X'] = test_set_x
            sets[test_set]['Y'] = test_set_y
            sets[test_set]['name'] = '3_test_%s' %(test_set)
            sets[test_set]['indices'] = []

    if rotation_file:
        with open(rotation_file, 'rb') as fp:
            rotation_lists = pickle.load(fp)
    rotation_set = None

    if merge_sets:
        sets['all_sets']['X'] = np.array(sets['all_sets']['X'])
        sets['all_sets']['Y'] = append_orderedDict(sets['all_sets']['Y'])
        set_names = ['all_sets']
    else:
        #set_names = Test[dataSet].keys()
        set_names = sets.keys()

    for sub_set in set_names:
        set_x = sets[sub_set]['X']
        set_y = sets[sub_set]['Y']
        set = sets[sub_set]['name']

        set_y_cp = deepcopy(set_y)
        set_x_cp = deepcopy(set_x)

        if max_errors:
            indices, error_kpt_avg_all = eval_test_set(tcdcn, params, set_x, set_y, set,
                                                       sample_num, dataSet, cfNet_model)
            set_x_indx, set_y_indx = get_indices(set_x_cp, set_y_cp, indices)
        elif indices_given:
            set_x_indx, set_y_indx = get_indices(set_x_cp, set_y_cp, sets[sub_set]['indices'])
        elif pickle_path:
            pickle_indices, all_vals = get_pickle_indices(pickle_path)
            set_x_indx, set_y_indx = get_indices(set_x_cp, set_y_cp, pickle_indices)
        else:
            set_x_indx, set_y_indx = get_subsets(set_x_cp, set_y_cp, sample_num)

        ##############################################
        # getting the 2nd bounding box test set data #
        ##############################################
        # preprocessing data
        rng_seed  = np.random.RandomState(0)
        scale_mul = 0.0
        translate_mul = 0.0
        target_dim =  params['target_dim']
        td = (target_dim, target_dim)
        if rotation_file and 'test' in set:
            rotation_list = np.array(rotation_lists[sub_set])
            rotation_set = rotation_list[indices]
        else:
            rotation_set = None

        set_y_cp = deepcopy(set_y_indx)
        set_x_cp = deepcopy(set_x_indx)

        if dataSet == 'MTFL':
            dist_ratio = 3.8/4.0
        else:
            dist_ratio = params['dist_ratio']

        set_x2, set_y2 = preprocess_once(set_x_indx, set_y_indx, dist_ratio=dist_ratio, gray_scale=params['gray_scale'])
        # using preprocess_iter to prepare set_x2 and set_y2 as the model expect it
        set_x2, set_y2 = preprocess_iter(set_x2, set_y2, rng_seed, jitter=False, scale_mul=scale_mul,
                                         translate_mul=translate_mul, target_dim=td, sanity=False, dset=dataSet,
                                         use_lcn=params['use_lcn'], rotation_set=rotation_set)
        # using preprocessing with gray_scale=False and use_lcn=True to get set_x_show in RGB format
        set_x_show, set_y_show = preprocess_once(set_x_cp, set_y_cp, dist_ratio=dist_ratio, gray_scale=False)
        set_x_show, _ = preprocess_iter(set_x_show, set_y_show, rng_seed, jitter=False, scale_mul=scale_mul,
                                        translate_mul=translate_mul, target_dim=td, sanity=False, dset=dataSet,
                                        use_lcn=True, rotation_set=rotation_set)
        # getting the keypoint results for the first images in the test set

        # face_rect has a vector of four values: rect_start_x, rect_start_y, rect_width (rect_end_x - rect_start_x), rect_height (rect_end_y - rect_start_y)
        set_rect_2nd = set_y2['face_rect']
        #getting the rectangle size for the detected face
        rect_size_2nd = set_rect_2nd[:, 2:]
        # getting the start point of the ractangle
        rect_start_point_2nd = set_rect_2nd[:, :2]
        # broadcasting the rectangle size and the start point by the number of the keypoints
        set_rect_size_2nd = np.tile(rect_size_2nd, (1, num_kpt))
        set_rect_start_2nd = np.tile(rect_start_point_2nd, (1, num_kpt))

        # the keypoint positions in the normalized format
        set_kpt_norm = set_y2['kpt_norm']

        # setting batch_size to one to avoid getting different errors for different batch sizes
        batch_size = 1
        num_batches = int(np.ceil(set_x2.shape[0]/float(batch_size)))
        epoch_error_kpt_avg = []
        kpt_conv = []
        for index in np.arange(num_batches):
            x_batch = set_x2[index * batch_size: (index + 1) * batch_size]
            if dataSet == 'MTFL':
                kpt_conv_batch = tcdcn.get_keypoints_MTFL(x_batch, dropout=0)
            else: # dataSet == '300W'
                if 'test' in set:
                    bound_mask = set_y2['bound_mask'][index * batch_size: (index + 1) * batch_size]
                    pad_ratio = set_y2['pad_ratio'][index * batch_size: (index + 1) * batch_size]
                    border_pixel = padRatio_to_pixels(pad_ratio, set_x2.shape[1])
                    if params['denoise_conv'] == 1:
                        one_hot_Maps = get_one_hot_predictions(tcdcn_cfNet, x_batch, params['target_dim'])
                        if mult_probs:
                            print "mult_probs code is not complete yet"
                        else:
                            kpt_conv_batch = tcdcn.get_keypoints_MTFL(one_hot_Maps, bound_mask, border_pixel, dropout=0)
                    else: # using coarse_fine_conv models
                        kpt_conv_batch = tcdcn.get_keypoints_MTFL(x_batch, bound_mask, border_pixel, dropout=0)

                else: # train and valid sets
                    if params['denoise_conv'] == 1:
                        one_hot_Maps = get_one_hot_predictions(tcdcn_cfNet, x_batch, params['target_dim'])
                        if mult_probs:
                            print "mult_probs code is not complete yet"
                        else:
                            kpt_conv_batch = tcdcn.get_keypoints_MTFL_train(one_hot_Maps, dropout=0)
                    else: # using coarse_fine_conv models
                        kpt_conv_batch = tcdcn.get_keypoints_MTFL_train(x_batch, dropout=0)

            kpt_conv.extend(kpt_conv_batch)
        kpt_conv = np.array(kpt_conv)

        if plot_colored_kpt:
            high_res = True
        if high_res:
            kpt_conv_shifted = (kpt_conv / float(target_dim)) * set_rect_size_2nd + set_rect_start_2nd
            kpt_conv_shifted = kpt_conv_shifted.astype(int)
            kpt_true_shifted = set_kpt_norm * set_rect_size_2nd + set_rect_start_2nd
            kpt_true_shifted = kpt_true_shifted.astype(int)
        else:
            kpt_conv_shifted = kpt_conv
            kpt_true_shifted = set_kpt_norm * target_dim

        n_samples = set_x_indx.shape[0]
        for i in xrange(n_samples):
            index = i+1
            if high_res:
                img = set_x_indx[i]
            else:
                img = set_x_show[i]
            if plot_colored_kpt:
                img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            img_kpts = drawpoints(img, kpt_conv_shifted[i], kpt_true_shifted[i], magnify=False, plot_colored_kpt=plot_colored_kpt)
            if annotate_kpts:
                img_kpts = annotateKpts(img_kpts ,kpt_conv_shifted[i], kpt_true_shifted[i])
            out_file_path = "%s/%s_%i_downsampled.png" %(out_path, set, (i+1))
            cv2.imwrite(out_file_path, img_kpts)

        # tiling the images
        print "tiling images"
        img = cv2.imread(out_file_path)
        row_size, col_size = img.shape[:2]
        tile_images(n_samples, set, out_path, row_size, col_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting the model keypoint predictions on image faces.')
    parser.add_argument('--path', type=str, help='the complete path to the model\'s pickle file', required=True)
    parser.add_argument('--cfNet_path', type=str, help='the complete path to the cfNet model\'s pickle file, whose output\
                        is taken as the input to the model given by path', default="")
    parser.add_argument('--max_errors', action='store_true', default=False)
    parser.add_argument('--high_res', action='store_false', default=True)
    parser.add_argument('--sample_num', type=int, default=20)
    parser.add_argument('--rotation_file', type=str, help='the complete path to the pickle file for rotations', default="")
    parser.add_argument('--plot_colored_kpt', action='store_true', default=False)
    parser.add_argument('--indices_given', action='store_true', default=False)
    parser.add_argument('--mult_probs', help='indicates to multiply the probabilities of two models', action='store_true', default=False)
    parser.add_argument('--merge_sets', help='indicates whether to merge the test sets as one set or not', action='store_true', default=False)
    parser.add_argument('--pickle_path', type=str, help='the complete path to the indices', default="")

    args = parser.parse_args()
    high_res = args.high_res
    indices_given = args.indices_given
    plot_colored_kpt = args.plot_colored_kpt
    # tha path to the folder that contains MTFL datasets
    pkl_param_file = args.path
    max_errors = args.max_errors
    sample_num = args.sample_num
    rotation_file = args.rotation_file
    cfNet_path = args.cfNet_path
    mult_probs = args.mult_probs
    merge_sets = args.merge_sets
    pickle_path = args.pickle_path

    print "max_errors is %s" %(max_errors,)

    # tha path to the folder that contains MTFL datasets
    image_path = face_path + '/datasets/MTFL_images'
    test_dir_aflw = "%s/AFLW" %(image_path)
    test_dir_afw = "%s/AFW" %(image_path)

    # getting the name of pkl_param_file
    out_parts = pkl_param_file.split('/')
    parent_dir = '/'.join(out_parts[:-1])
    assert parent_dir is not None

    # creating the base output dir
    base_out_path = "%s/detected_kpts" %(parent_dir)
    if not os.path.exists(base_out_path):
        os.makedirs(base_out_path)

    # creating the secondary output dir
    pkl_name = pkl_param_file.split('.')
    pkl_name = '.'.join(pkl_name[:-1])
    if 'params_epoch' in pkl_name:
        pkl_name = pkl_name.split('params_epoch')[1]
        pkl_name = pkl_name.split('_')
        pkl_name = '_'.join(pkl_name[2:])
    else:
        pkl_name = pkl_name.split('params_')[1]
    out_path = "%s/%s" %(base_out_path, pkl_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    draw_points_raw(out_path, annotate_kpts=False, max_errors=max_errors, sample_num=sample_num,
                    high_res=high_res, plot_colored_kpt=plot_colored_kpt, indices_given=indices_given,
                    cfNet_path=cfNet_path, mult_probs=mult_probs, merge_sets=merge_sets,
                    pickle_path=pickle_path)
