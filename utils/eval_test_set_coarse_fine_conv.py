# This module takes the weights of a keypoint detection convnet and
# sets the convnet weights to them, then it passes to the convnet
# the images from the sets and gets the error_kpt_avg on them.

import numpy as np
import sys
from RCN.models.create_procs import get_data
from RCN.preprocessing.tools import shuffleData, splitData
from RCN.preprocessing.preprocess import preprocess_iter, preprocess_once
from RCN.utils.convnet_tools import create_TCDCN_obejct, get_error_kpt_avg
from collections import OrderedDict
import RCN
import argparse
import cv2
import os
from PIL import Image
import cPickle as pickle
face_path = os.path.dirname(RCN.__file__)
import string

def resize_img(img, size):
   # the methods for downsampling
   return img.resize(size, Image.ANTIALIAS)

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

def sort_arrays(array):
    array_asc = np.sort(array)[::-1]
    array_asc_indx = np.argsort(array)[::-1]
    return array_asc, array_asc_indx

def save_error_results(array, set_name):
    out_str_path = "%s/%s_sorted_error.txt" %(out_path, set_name)
    out_str = open(out_str_path,'w')
    out_str.write("error, index\n")

    array_asc, array_asc_indx = sort_arrays(array)

    for err, indx in zip(array_asc, array_asc_indx):
        out_str.write("%s, %s\n" %(err, indx))

    err_kpt = array

    out_str_path = "%s/%s_sorted_indices.pickle" %(out_path, set_name)
    with open(out_str_path, 'wb') as fp:
        pickle.dump(err_kpt, fp)
    print "done with %s" %(set_name)

def eval_test_sets(pkl_param_file, cfNet_path, mult_probs, use_batch_size_1,
                   joint_iterations=1, struct_iterations=0, use_previous_iter_pre_softmax=False):
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

    for test_set in Test[dataSet].keys():
        test_set_x, test_set_y = Test[dataSet][test_set]
        sets[test_set] = OrderedDict()
        sets[test_set]['X'] = test_set_x
        sets[test_set]['Y'] = test_set_y
        sets[test_set]['indices'] = []
        sets[test_set]['name'] = '3_test_%s' %(test_set)

    if rotation_file:
        with open(rotation_file, 'rb') as fp:
            rotation_lists = pickle.load(fp)
    rotation_set = None

    if use_batch_size_1:
        batch_size = 1
    else:
        batch_size = params['batch_size']

    #for sub_set in Test[dataSet].keys():
    for sub_set in sets.keys():
        set_x = sets[sub_set]['X']
        set_y = sets[sub_set]['Y']
        set_name = sub_set

        batch, row, col, ch = set_x.shape
        first_img_size = row

        # preprocessing data
        rng_seed  = np.random.RandomState(0)
        target_dim =  params['target_dim']
        td = (target_dim, target_dim)
        scale_mul = 0.0
        translate_mul = 0.0

        if rotation_file:
            rotation_set = rotation_lists[set_name]
        else:
            rotation_set = None

        if dataSet == 'MTFL':
            dist_ratio = 3.8/4.0
        else:
            dist_ratio = params['dist_ratio']

        set_x2, set_y2 = preprocess_once(set_x, set_y, dist_ratio=dist_ratio, gray_scale=params['gray_scale'])
        # using preprocess_iter to prepare set_x2 and set_y2 as the model expect it
        set_x2, set_y2 = preprocess_iter(set_x2, set_y2, rng_seed, jitter=False, scale_mul=scale_mul, translate_mul=translate_mul,
                     target_dim=td, sanity=False, dset=dataSet, use_lcn=params['use_lcn'], rotation_set=rotation_set)

        print "set name is %s" % set_name
        error_kpt_avg_all, error_kpt_avg = get_error_kpt_avg(tcdcn, params, set_x2, set_y2,
                                                             dataSet, set_name, cfNet_model,
                                                             mult_probs, batch_size=batch_size,
                                                             joint_iterations=joint_iterations,
                                                             struct_iterations=struct_iterations,
                                                             use_previous_iter_pre_softmax=use_previous_iter_pre_softmax)
        out_str.write("\n\n%s set error is %s\n" %(set_name, error_kpt_avg))
        save_error_results(error_kpt_avg_all, set_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting the model keypoint predictions on image faces.')
    parser.add_argument('--path', type=str, help='the complete path to the model\'s pickle file', required=True)
    parser.add_argument('--cfNet_path', type=str, help='the complete path to the cfNet model\'s pickle file, whose output\
                        is taken as the input to the model given by path', default="")
    parser.add_argument('--rotation_file', type=str, help='the complete path to the pickle file that contains rotation degree for each image', default="")
    parser.add_argument('--mult_probs', help='indicates to multiply the probabilities of two models', action='store_true', default=False)
    parser.add_argument('--use_batch_size_1', help='indicates to use only batch size one to get the errors', action='store_true', default=False)
    args = parser.parse_args()
    # tha path to the folder that contains MTFL datasets
    pkl_param_file = args.path
    rotation_file = args.rotation_file
    cfNet_path = args.cfNet_path
    mult_probs = args.mult_probs
    use_batch_size_1 = args.use_batch_size_1

    # getting the name of pkl_param_file
    out_parts = pkl_param_file.split('/')
    parent_dir = '/'.join(out_parts[:-1])
    assert parent_dir is not None

    out_dir_name = 'error_on_sets'
    base_out_path = "%s/%s" %(parent_dir, out_dir_name)
    if not os.path.exists(base_out_path):
        os.makedirs(base_out_path)

    subdir_name = out_parts[-1].split('shared_conv_params_')[1]
    subdir_name = subdir_name.split('.pickle')[0]
    subdir_out_path = "%s/%s" %(base_out_path, subdir_name)
    if not os.path.exists(subdir_out_path):
        os.makedirs(subdir_out_path)
    out_path = subdir_out_path

    out_path_str = "%s/test_set_results.txt" %(out_path)
    out_str = open(out_path_str,'w')
    eval_test_sets(pkl_param_file, cfNet_path, mult_probs, use_batch_size_1)
