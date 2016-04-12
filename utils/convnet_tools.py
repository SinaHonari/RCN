import numpy as np
import theano.tensor as T
import string
import sys
import time
import heapq
import math
import random
import cPickle as pickle
from RCN.preprocessing.tools import EOF, padRatio_to_pixels
from RCN.models.create_procs import get_nkerns, get_target_dim
from RCN.preprocessing.tools import (get_bound_mask, mask_padded_kpts,
                                            discretise_y, get_one_hot_map)

def set_one_for_nonzero(x):
    """Replaces nonzeros with ones

    This method gets a tensor and replaces non-zeros with ones such that
    the returned tensor is composed of only zeros and ones.

    """
    non_zero = (x > 0) + (x < 0)
    return non_zero * T.ones_like(x)

def softmax(x):
    """
    This method gets the softmax of an array (as vector, matrix, etc.)
    Getting the softmax over all of the variable
    """
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def softmax_2D(matrix):
    """
    This method gets softmax for each row of matrix
    """
    sh = matrix.shape[0]
    mx = np.max(matrix, axis=1).reshape(sh,1)
    x_minus_mx = matrix - mx
    e_matrix = np.exp(x_minus_mx)
    sum_row = np.sum(e_matrix, axis=1).reshape(sh, 1)
    out = e_matrix / sum_row
    return out

def not_normalized_sampling(weights, m):
    """Does not normalized sampling without replacement

    This method takes a set of weights which is not normalized
    and then takes m sample from it without replacement.

    parameters:
    ----------
    weights: a 1d vector
        a set of weights associated to different elements

    m: int
        the number of samples to be taken from weights.
        m <= len(weights)
    """
    elt = [(math.log(random.random()) / weights[i], i) for i in range(len(weights))]
    return [x[1] for x in heapq.nlargest(m, elt)]

def project_kpts_to_border(x_pred, y_pred, num_kpts, border_pixel):
    """
    This method project keypoints to border.

    parameters:
    ----------
        x_pred, y_pred: 1D vectros of dim '#batch * #kpt'

        num_kpts: the number of kpts

        border_pixel: a matrix of shape (#batch, 4),
                      indicating for each batch_example what are the 4 border_pixels

    returns:
    -------
        the masked x and y kpt locations of dim '#batch * #kpt'
    """
    x_pred_2D = x_pred.reshape(-1, num_kpts)
    y_pred_2D = y_pred.reshape(-1, num_kpts)

    # given the border_pixel, the predicted x and y should not be outside the bounds
    x_inside_all = 1. - (np.less(x_pred_2D, border_pixel[:, 0].reshape((-1, 1))) +\
                         np.greater(x_pred_2D, border_pixel[:, 2].reshape((-1, 1))))
    # x_pred_bord_all is of shape (#batch, #kpt)
    x_pred_bord_all = np.less(x_pred_2D, border_pixel[:, 0].reshape((-1, 1))) * border_pixel[:, 0].reshape((-1, 1)) +\
                      np.greater(x_pred_2D, border_pixel[:, 2].reshape((-1, 1))) * border_pixel[:, 2].reshape((-1, 1)) +\
                      x_inside_all * x_pred_2D
    y_inside_all = 1. - (np.less(y_pred_2D, border_pixel[:, 1].reshape((-1, 1))) +\
                         np.greater(y_pred_2D, border_pixel[:, 3].reshape((-1, 1))))
    # y_pred_bord_all is of shape (#batch, #kpt)
    y_pred_bord_all = np.less(y_pred_2D, border_pixel[:, 1].reshape((-1, 1))) * border_pixel[:, 1].reshape((-1, 1)) +\
                      np.greater(y_pred_2D, border_pixel[:, 3].reshape((-1, 1))) * border_pixel[:, 3].reshape((-1, 1)) +\
                      y_inside_all * y_pred_2D

    x_pred_masked = x_pred_bord_all.flatten()
    y_pred_masked = y_pred_bord_all.flatten()

    return x_pred_masked, y_pred_masked

def get_error_mult_probs_with_iteration(tcdcn, tcdcn_cfNet, x,
                                        one_hot_Maps, dim, y_kpt_MTFL,
                                        y_kpt_ocular_dist, num_kpts,
                                        border_pixel, set_kpts_to_border=True,
                                        use_pre_softmax=False, joint_iterations=1,
                                        struct_iterations=0,
                                        use_previous_iter_pre_softmax=False):

    # getting pre_softmax maps of cfNet model
    pre_softmax_maps = tcdcn_cfNet.get_pre_softmax(x, dropout=0)

    for iter in np.arange(joint_iterations):
        if iter == joint_iterations - 1:
            # in the last iteration the keypoint are mapped to border
            set_kpts_to_border_iter = set_kpts_to_border
        else:
            set_kpts_to_border_iter = False

        for struc_iter in np.arange(struct_iterations):
            # in this case, the one_hot_Maps are passed
            # struct_iterations times into the tcdcn (strucured model)
            # and it is asked to denoise its predictions.
            one_hot_Maps = get_one_hot_predictions(tcdcn, one_hot_Maps, dim)

        error_kpt_avg, predict_2D, sum_pre_softmax_maps = get_error_mult_probs(
                                 tcdcn=tcdcn, x=x, one_hot_Maps=one_hot_Maps,
                                 dim=dim, y_kpt_MTFL=y_kpt_MTFL,
                                 y_kpt_ocular_dist=y_kpt_ocular_dist,
                                 num_kpts=num_kpts, border_pixel=border_pixel,
                                 set_kpts_to_border=set_kpts_to_border_iter,
                                 pre_softmax_cfNet=pre_softmax_maps,
                                 use_pre_softmax=use_pre_softmax)

        # setting the one_hot_Maps to the one_hot prediciotn of the joint
        # model in the previous iteration
        one_hot_Maps = make_2D_predictions_into_one_hot_4D(predict_2D, dim)

        if use_previous_iter_pre_softmax:
            # in this case, from the 2nd iteration forward use the sum of pre_softmax
            # maps instead of the cfNet_softmax_maps
            pre_softmax_maps = sum_pre_softmax_maps

    return error_kpt_avg

def get_error_mult_probs(tcdcn, x, one_hot_Maps, dim, y_kpt_MTFL,
                         y_kpt_ocular_dist, num_kpts, border_pixel,
                         pre_softmax_cfNet, set_kpts_to_border=True,
                         use_pre_softmax=False):
    """
    This module gets two models, gets the pre-softmax values of them
    sums them together, and pass the result through a softmax, which
    is equivalent to multiplying both probs. It then gets the max values
    of the softmax, and gets the euclidean distance between true and
    predicted key-point positions, normalized by inter-ocular distance.

    returns:
    ---------
    The average error for each example over all keypoints.
    """
    if use_pre_softmax:
        pre_softmax_struc = tcdcn.get_pre_softmax(one_hot_Maps, pre_softmax_cfNet, dropout=0)
    else:
        pre_softmax_struc = tcdcn.get_pre_softmax(one_hot_Maps, dropout=0)
    # sum_softmax is of shape (#batch, #kpts, #dim ,#dim)
    sum_softmax = pre_softmax_cfNet + pre_softmax_struc
    bch, nkpts, row, col = sum_softmax.shape
    sum_softmax_2D = sum_softmax.reshape(bch * nkpts, row * col)
    softmax = softmax_2D(sum_softmax_2D)
    # predict is of shape (#batch * #kpt)
    predict = np.argmax(softmax, axis=1)
    # y_pred is of shape (#batch * #kpt)
    y_pred = predict // dim
    # x_pred is of shape (#batch * #kpt)
    x_pred = predict % dim

    if set_kpts_to_border:
        x_pred, y_pred = project_kpts_to_border(x_pred, y_pred, num_kpts, border_pixel)

    y_kpt_norm_serial = y_kpt_MTFL.flatten()
    # y_true is of shape (#batch * #kpt)
    y_true = y_kpt_norm_serial // dim
    # x_true is of shape (#batch * #kpt)
    x_true = y_kpt_norm_serial % dim
    x_diff_sqr = (x_pred - x_true)**2
    y_diff_sqr = (y_pred - y_true)**2
    # kpt_euc_dist is of shape (#batch * #kpt)
    kpt_euc_dist = np.sqrt(x_diff_sqr + y_diff_sqr)
    # error_kpt_2D is of shape (#batch, #kpt)
    error_kpt_2D = kpt_euc_dist.reshape(-1, nkpts)
    y_kpt_ocular_unorm = y_kpt_ocular_dist * dim
    error_kpt_each_norm = error_kpt_2D / y_kpt_ocular_unorm.reshape(bch, 1)
    # error_kpt_avg is of dim #batch
    error_kpt_avg = np.mean(error_kpt_each_norm, axis=1)

    predict_2D = predict.reshape(bch, nkpts)
    return error_kpt_avg, predict_2D, sum_softmax

def make_2D_predictions_into_one_hot_4D(prediction_2D, dim):
    """
    This method gets 2D prediction of shape (#batch, #kpts)
    and then returns 4D one_hot maps of shape
    (#batch, #kpts, #dim, #dim)
    """

    # getting one_hot maps of predicted locations
    # one_hot_maps is of shape (#batch, #kpts, #dim * #dim)
    one_hot_Maps = get_one_hot_map(prediction_2D, dim)

    num_batch, num_kpt = prediction_2D.shape
    one_hot_Maps_4D = one_hot_Maps.reshape(num_batch, num_kpt, dim, dim)
    return one_hot_Maps_4D

def get_one_hot_predictions(tcdcn, x, dim):
    """
    This method gets a model (tcdcn), passes x through it
    and gets it's prediction, then it gets one_hot
    matrix representation of the predictions. depending on whether
    tcdcn is RCN or a structured model, x can be an image
    (in the former case) and a one-hot map representation
    (in the latter case).
    """
    # prediction_2D is a matrix of shape (#batch, #kpts), with
    # each value in the range [0, dim**2)
    prediction_2D = tcdcn.model_prediction(x, dropout=0)

    # getting 4D one_hot maps from 2D predictions
    one_hot_Maps_4D = make_2D_predictions_into_one_hot_4D(prediction_2D, dim)
    return one_hot_Maps_4D

def create_TCDCN_obejct(pkl_param_file, dset='MTFL', create_object=True):

    if 'params_epoch' in pkl_param_file:
        start_suffix = 5 # the index of the first element in the pickle file that has the file suffix
    else:
        start_suffix = 3
    path = pkl_param_file.split('/')
    name_suffix = path[-1].split('_')[start_suffix:]
    if dset == 'MTFL' or dset == '300W':
        setting_name = 'shared_conv_setting_' + '_'.join(name_suffix[:-1]) + '.pickle'
    else:
        setting_name = 'shared_conv_setting_' + '_'.join(name_suffix)
    path[-1] = setting_name
    setting_path = '/'.join(path)
    with open(setting_path, 'rb') as fp:
        params = pickle.load(fp)

    cost = params['cost']
    gray_scale = params['gray_scale']
    mask_MTFL = params['mask_MTFL']
    mask_300W = params['mask_300W']
    if gray_scale:
        num_img_channels = 1
    else:
        num_img_channels = 3

    if 'denoise_conv' not in params.keys():
        params['denoise_conv'] = 0.

    nkerns = get_nkerns(**params)

    paral_conv = params['paral_conv']
    denoise_conv = params['denoise_conv']
    if paral_conv:
        if paral_conv == 1:
            from RCN.models.SumNet_MTFL import TCDCN_ConvNet
        elif paral_conv == 2:
            from RCN.models.SumNet_300W import TCDCN_ConvNet
        elif paral_conv == 3:
            from RCN.models.RCN_MTFL import TCDCN_ConvNet
        elif paral_conv == 4:
            from RCN.models.RCN_MTFL_skip import TCDCN_ConvNet
        elif paral_conv == 5:
            from RCN.models.RCN_300W import TCDCN_ConvNet
        elif paral_conv == 6:
            from RCN.models.RCN_300W_skip import TCDCN_ConvNet
    elif params['denoise_conv']:
        denoise_conv = params['denoise_conv']
        if denoise_conv == 1:
            from RCN.models.Denoising_300W import TCDCN_ConvNet
        elif denoise_conv == 2:
            from RCN.models.fine_tune_cfNet_structured import TCDCN_ConvNet
            # loading coarse-fine convnet model
            print "loading cfNet params"
            _, hp_params_cfNet = create_TCDCN_obejct(params['param_path_cfNet'], create_object=False)
            params['hp_params_cfNet'] = hp_params_cfNet
            params['nkerns_cfNet'] = get_nkerns(**hp_params_cfNet)
            # loading structured convnet model
            print "loading struc params"
            _, hp_params_struc = create_TCDCN_obejct(params['param_path_strucNet'], create_object=False)
            params['hp_params_struc'] = hp_params_struc
            params['nkerns_struc'] = get_nkerns(**hp_params_struc)
            params['num_keypoints'] = 68
            params['num_img_channels'] = num_img_channels
            print "loaded params"

    if 'use_softmax_maps' in params.keys():
        use_soft_maps = params['use_softmax_maps']
        use_soft_maps = use_soft_maps.split(',')
        use_soft_maps = map(string.strip, use_soft_maps)
        use_soft_maps = map(int, use_soft_maps)
        assert all(x==0 or x==1 for x in use_soft_maps)
        use_softmax_maps = np.array(use_soft_maps)
        params['use_softmax_maps'] = use_softmax_maps

    if 'mask_branch' in params.keys():
        br_mask = params['mask_branch']
        br_mask = br_mask.split(',')
        br_mask = map(string.strip, br_mask)
        br_mask = map(int, br_mask)
        assert all(x==0 or x==1 for x in br_mask)
        mask_branch = np.array(br_mask)
    else:
        mask_branch = np.array([1,1,1,1])
    params['mask_branch'] = mask_branch

    if 'coarse_mask_branch' in params.keys():
        br_mask = params['coarse_mask_branch']
        br_mask = br_mask.split(',')
        br_mask = map(string.strip, br_mask)
        br_mask = map(int, br_mask)
        assert all(x==0 or x==1 for x in br_mask)
        coarse_mask_branch = np.array(br_mask)
    else:
        coarse_mask_branch = np.array([1,1,1,1,1])
    params['coarse_mask_branch'] = coarse_mask_branch

    target_dim = params['target_dim']
    bilinear = params['bilinear']
    target_dim = get_target_dim(target_dim, bilinear)
    params['target_dim'] = target_dim

    if create_object:
        if paral_conv:
            tcdcn = TCDCN_ConvNet(train_cost=cost, num_img_channels=num_img_channels,
                                  mask_300W_layer=mask_300W, mask_MTFL_layer=mask_MTFL,
                                  nkerns=nkerns, **params)
        elif params['denoise_conv']:
            rng = np.random.RandomState(params['param_seed'])
            tcdcn = TCDCN_ConvNet(nkerns=nkerns, rng=rng, **params)
    else:
        tcdcn = None
    return [tcdcn, params]


def get_error_kpt_avg(tcdcn, params, test_set_x, test_set_y, dataSet, set_name,
                      cfNet_model=(None, None), mult_probs=False,
                      batch_size=None, joint_iterations=1, struct_iterations=0,
                      use_previous_iter_pre_softmax=False):
    # getting cfNet model convnet and params
    tcdcn_cfNet, params_cfNet = cfNet_model
    if not batch_size:
        batch_size = params['batch_size']
    test_num_batches = int(np.ceil(test_set_x.shape[0]/float(batch_size)))
    test_num_samples = test_set_x.shape[0]
    error_kpt_avg_all = []
    kpt_norm = test_set_y['kpt_norm']

    if dataSet == '300W':
        ###########################
        # getting the mask_border #
        ###########################
        bound_mask = test_set_y['bound_mask']
        mask_border = get_bound_mask(bound_mask)
        mask_border = mask_padded_kpts(kpt_norm, mask_border)
        test_set_y['mask_border'] = mask_border
        #########################
        # getting border_pixels #
        #########################
        pad_ratio = test_set_y['pad_ratio']
        border_pixel = padRatio_to_pixels(pad_ratio, test_set_x.shape[1])
        test_set_y['border_pixel'] = border_pixel

    dim = params['target_dim']
    kpt_discret = discretise_y(kpt_norm, dim)
    test_set_y['kpt_norm'] = kpt_discret

    start_time = time.time()
    for index in np.arange(test_num_batches):
        x = test_set_x[index * batch_size: (index + 1) * batch_size]
        y_kpt_MTFL = test_set_y['kpt_norm'][index * batch_size: (index + 1) * batch_size]
        y_kpt_ocular_dist = test_set_y['ocular_dist'][index * batch_size: (index + 1) * batch_size]

        if dataSet == 'MTFL':
            # in this case only the convnet is applied to the MTFL dataset
            mask_MTFL = np.ones((x.shape[0]))
            error_kpt_avg = tcdcn.get_errors(x, y_kpt_ocular_dist, y_kpt_MTFL, mask_MTFL, dropout=0)
        else: # dataSet == '300W'
            if set_name in ['ibug', 'lfpw', 'Helen']:
                bound_mask = test_set_y['bound_mask'][index * batch_size: (index + 1) * batch_size]
                border_pixel = test_set_y['border_pixel'][index * batch_size: (index + 1) * batch_size]
                if params['denoise_conv'] == 1:
                    one_hot_Maps = get_one_hot_predictions(tcdcn_cfNet, x, dim)
                    use_pre_softmax = False
                    if mult_probs:
                        # It passes the predictions of the first model to the second model and in addition it sums the pre-softmax
                        # values of both models and pass it through a softmax to get the final prediction
                        error_kpt_avg = get_error_mult_probs_with_iteration(tcdcn, tcdcn_cfNet, x, one_hot_Maps,
                                                                            dim, y_kpt_MTFL, y_kpt_ocular_dist,
                                                                            num_kpts=params['num_model_kpts'], border_pixel=border_pixel,
                                                                            set_kpts_to_border=True, use_pre_softmax=use_pre_softmax,
                                                                            joint_iterations=joint_iterations,
                                                                            struct_iterations=struct_iterations,
                                                                            use_previous_iter_pre_softmax=use_previous_iter_pre_softmax)
                    else:
                        # using one model as the input to another model, which passes the predictions of the first model to the second model
                        # using get_error of structured_kpt_dist model
                        # first get one_hot_map of the cfNet model's prediction
                        mask_kpts = np.ones_like(y_kpt_MTFL).flatten()
                        error_kpt_avg = tcdcn.get_errors(one_hot_Maps, y_kpt_ocular_dist, y_kpt_MTFL,
                                                         bound_mask, border_pixel, mask_kpts, dropout=0)
                elif params['denoise_conv'] == 2:
                    mask_kpts = np.ones_like(y_kpt_MTFL).flatten()
                    error_kpt_avg = tcdcn.get_errors(x, y_kpt_ocular_dist, y_kpt_MTFL, border_pixel, mask_kpts, dropout=0)
                else:
                    # using get_error of chained model
                    error_kpt_avg = tcdcn.get_errors(x, y_kpt_ocular_dist, y_kpt_MTFL, bound_mask, border_pixel, dropout=0)
            else:
                if params['denoise_conv'] == 1:
                    one_hot_Maps = get_one_hot_predictions(tcdcn_cfNet, x, dim)
                    use_pre_softmax = False
                    if mult_probs:
                        # It passes the predictions of the first model to the second model and in addition it sums the pre-softmax
                        # values of both models and pass it through a softmax to get the final prediction
                        error_kpt_avg = get_error_mult_probs_with_iteration(tcdcn, tcdcn_cfNet, x, one_hot_Maps,
                                                                            dim, y_kpt_MTFL, y_kpt_ocular_dist,
                                                                            num_kpts=params['num_model_kpts'], border_pixel=None,
                                                                            set_kpts_to_border=False, use_pre_softmax=use_pre_softmax,
                                                                            joint_iterations=joint_iterations,
                                                                            struct_iterations=struct_iterations,
                                                                            use_previous_iter_pre_softmax=use_previous_iter_pre_softmax)
                    else:
                        # using one model as the input to another model, which passes the predictions of the first model to the second model
                        # using get_error of structured_kpt_dist model
                        # first get one_hot_map of the cfNet model's prediction
                        mask_kpts = np.ones_like(y_kpt_MTFL).flatten()
                        error_kpt_avg = tcdcn.get_errors_train(one_hot_Maps, y_kpt_ocular_dist, y_kpt_MTFL, mask_kpts, dropout=0)
                elif params['denoise_conv'] == 2:
                    mask_kpts = np.ones_like(y_kpt_MTFL).flatten()
                    error_kpt_avg = tcdcn.get_errors_train(x, y_kpt_ocular_dist, y_kpt_MTFL, mask_kpts, dropout=0)
                else:
                    # using get_error of chained model
                    mask_border = test_set_y['mask_border'][index * batch_size: (index + 1) * batch_size]
                    mask_border = np.ndarray.flatten(mask_border)
                    error_kpt_avg = tcdcn.get_errors_train(x, y_kpt_ocular_dist, y_kpt_MTFL, mask_border, dropout=0)

        error_kpt_avg_all.extend(error_kpt_avg)

    end_time = time.time()
    run_time = (end_time - start_time)
    sys.stderr.write('test time took %f minutes\n' % (run_time / 60.))
    error_kpt_avg_mean = np.sum(error_kpt_avg_all)/test_num_samples
    return error_kpt_avg_all, error_kpt_avg_mean
