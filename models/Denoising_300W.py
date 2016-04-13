"""
This model is the denoising auto-encoder submitted to CVPR,
trained on one-hot true key-point positions, half of which is jittered
in training.
"""
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from collections import OrderedDict
import cPickle as pickle
import time
import sys
import random
sys.setrecursionlimit(2000)
import RCN
from RCN.utils.grad_updates import Train_alg
from RCN.preprocessing.tools import (EOF, padRatio_to_pixels,
                                            discretise_y, mask_padded_kpts,
                                            get_bound_mask, get_one_hot_map)
from RCN.preprocessing.local_contrast_normalization import lcn
from RCN.utils.convnet_tools import create_TCDCN_obejct, get_one_hot_predictions
from RCN.models.layers import (ConvPoolLayer, Softmax)
import os

source_dir = os.path.dirname(RCN.__file__)
dest_dir = source_dir + '/models/exp_shared_conv'

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


def get_source_points(tcdcn_cfNet, x, y_kpt_norm, num_model_kpts, use_tcdcn=False):
    """
    This method gets the source keypoints for training the model.
    Either a model's output or the true keypoints or a combination
    of two is taken as the source.

    paramers:
        num_model_kpts: the number of kpts to be used from cfNet model per example
                        default is all kpts
    returns: The keypoint locations upon which the model should
             train to denoise the output. It's a matrix of shape (#batch, #kpts)
    """
    # source_points is of shape (#batch, #kpts)
    if use_tcdcn:
        model_points = tcdcn_cfNet.model_prediction(x, dropout=0)
        num_batch, num_kpts = y_kpt_norm.shape
        source_points = model_points

        """
        if num_model_kpts != num_kpts:# default is 68 for 300W dataset
            # in this case the true and predicted keypoints should be merged
            # get as many as 'num_model_kpts' random kpts per batch example
            ind_0, ind_1 = get_rnd_kpts_per_samples(num_batch, num_kpts, num_model_kpts)
            source_points = y_kpt_norm
            # replacing true values with the model's prediction
            source_points[ind_0, ind_1] = model_points[ind_0, ind_1]
        else:
            # in this case a model's output is taken as the source
            source_points = model_points
        """
    else:
        # in this case the true keypoint positions is taken as the source
        source_points = y_kpt_norm
    return source_points


def get_and_noise_one_hot_maps(kpts, dim, nMaps_shuffled, rng, dropout_kpts):
    """
    This method gets the one-hot vectors for the true-kpt locations
    selects random kpts in each example and replaces the true position
    with a different random one-hot representation.

    parameters:
        kpts: a matrix of shape (#batch, #kpts)
              with each value in the range of
              [0, dim**2)
        dim: the dimension (row or col) of the target feature_maps
        nMaps_shuffled: the number of kpts that are jittered in each example
        rng: numpy random state for drawing seeded random values
        dropout_kpts: a boolean indicating whether to completely mask the random
                   selected keypoints or not. if true, the maps corresponding to
                   those keypoints are all zeros.
    returns:
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

    ########################################
    # generating random locations for kpts #
    ########################################
    ind_0, ind_1 = get_rnd_kpts_per_samples(num_batch, num_kpts, nMaps_shuffled)

    # setting selected kpt indices in one_hot_maps to zero
    one_hot_maps[ind_0, ind_1, :] = 0

    if not dropout_kpts:
        # building the dimension two of indices
        # getting random int number in the range [0, dim**2)
        # for random keypoint position genration
        ind_2 = np.random.randint(low=0, high=dim*dim, size=nMaps_shuffled * num_batch)

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

def shuffle_heatMaps(hMaps, nMaps_shuffled, rng):
    """
    This method gets the heatmaps generated by the first convnet,
    selects randomly as many as 'nMaps_shuffled' heatMaps from each
    batch sample and shuffles all selected heatMaps (from all sampels)
    and finally replaces them with the random shuffled ones.

    parameters:
        hMaps: a 4D tensor of shape (#batch, #kpts, #rows, #cosl)
               whose some of heatMaps will be shuffled and replaced.
               randomly with other heatMaps
        nMaps_shuffled: an integer indicating the number of
                      heatMaps in each batch eample to be replaced.
    """
    num_batch, num_kpts = hMaps.shape[:2]
    # getting random int values for the keypoint-heatMaps to be replaced
    rand_hMap_idx = rng.randint(low=0, high=num_kpts, size=nMaps_shuffled * num_batch)

    # we select as many as 'nMaps_shuffled' heatMaps from each mini_batch sample
    # so indices 0 to num_batch, are each repeated as many as 'nMaps_shuffled'
    range_idx = np.arange(num_batch)
    rand_idx = np.repeat(range_idx, nMaps_shuffled)

    # getting the randomly selected heatMaps
    # rand_hMaps is of shape (nMaps_shuffled * num_batc, #rows, #cols)
    rand_hMaps = hMaps[rand_idx, rand_hMap_idx]

    # shuffling the rand_hMaps along its first dimension
    # rand_hMaps is of shape (nMaps_shuffled * num_batc, #rows, #cols)
    rng.shuffle(rand_hMaps)

    hMaps[rand_idx, rand_hMap_idx] = rand_hMaps
    return hMaps

def get_heatMaps(tcdcn_cfNet, x):
    return tcdcn_cfNet.get_softmax_output(x, dropout=0)

def get_model_prediction(tcdcn_cfNet, x):
    """
    This method gets prediction of the model, with
    each value in the range [0, dim**2)
    """
    return tcdcn_cfNet.get_softmax_output(x, dropout=0)

def get_and_shuffle_heatMaps(tcdcn_cfNet, x, nMaps_shuffled, rng):
    # getting the heat-Maps generated by first convnet
    hMaps = get_heatMaps(tcdcn_cfNet, x)

    # shuffling the heat-Maps generated by fist convnet
    if nMaps_shuffled > 0:
        hMaps_shuffled = shuffle_heatMaps(hMaps, nMaps_shuffled, rng)
        return hMaps_shuffled
    else:
        return hMaps


def fforward_model(layer1_input, dropout, nkerns, conv_size, num_img_channels, rng, dim=80):
    ############################
    # building the conv layers #
    ############################
    conv_border = conv_size / 2
    conv_small = 1

    bch_norm = False
    #####################
    ### hidden layer1 ###
    #####################
    layerSh1 = ConvPoolLayer(
        rng,
        dropout = dropout,
        bch_norm = bch_norm,
        input=layer1_input,
        image_shape=(None, num_img_channels, None, None),
        filter_shape=(nkerns[1], num_img_channels, conv_size, conv_size),
        act='relu',
        pool=False,
        border_mode=(conv_border, conv_border)
    )
    layerSh1_output = layerSh1.output
    # layerSh1_output is now of shape (#batch, #nkerns[1]=64, 80, 80)
    layerSh1.W.name = 'Conv_layerSh0_W'
    layerSh1.b.name = 'Conv_layerSh0_b'
    layerSh1 = layerSh1

    ######################
    ### hidden layers2 ###
    ######################
    # shared conv layer
    layerSh2 = ConvPoolLayer(
        rng,
        dropout = dropout,
        bch_norm = bch_norm,
        input=layerSh1_output,
        image_shape=(None, nkerns[1], None, None),
        filter_shape=(nkerns[2], nkerns[1], conv_size, conv_size),
        act='relu',
        pool=False,
        border_mode=(conv_border, conv_border)
    )
    layerSh2_output = layerSh2.output
    # layerSh2_output is now of shape (#batch, #nkerns[2]=64, 80, 80)
    layerSh2.W.name = 'Conv_layerSh2_W'
    layerSh2.b.name = 'Conv_layerSh2_b'
    layerSh2 = layerSh2

    ######################
    ### hidden layers3 ###
    ######################
    # shared conv layer
    layerSh3 = ConvPoolLayer(
        rng,
        dropout = dropout,
        bch_norm = bch_norm,
        input=layerSh2_output,
        image_shape=(None, nkerns[2], None, None),
        filter_shape=(nkerns[3], nkerns[2], conv_size, conv_size),
        act='relu',
        pool=False,
        border_mode=(conv_border, conv_border)
    )
    layerSh3_output = layerSh3.output
    # layerSh3_output is now of shape (#batch, #nkerns[2]=64, 80, 80)
    layerSh3.W.name = 'Conv_layerSh3_W'
    layerSh3.b.name = 'Conv_layerSh3_b'
    layerSh3 = layerSh3

    #####################
    ### output layers ###
    #####################
    layerFOut = ConvPoolLayer(
        rng,
        dropout = dropout,
        bch_norm = bch_norm,
        input=layerSh3_output,
        image_shape=(None, nkerns[3], None, None),
        filter_shape=(nkerns[0], nkerns[3], conv_small, conv_small),
        act='linear',
        pool=False,
        border_mode='valid'
    )
    layerFOut.W.name = 'Conv_layerOut_W'
    layerFOut.b.name = 'Conv_layerOut_b'
    layerFOut = layerFOut
    layerFOut_output = layerFOut.output
    # layerOut_output is now of shape (#batch, #nkerns[0]=68, 80, 80)

    ##############################
    # buildign the softmax layer #
    ##############################
    # conv_sum is of shape (#batch_size, #kpts, #rows, #cols)
    conv_sum = layerFOut_output

    # conv_sum_3D is of shape (#batch_size * #kpts, #rows, #cols)
    conv_sum_3D = conv_sum.reshape((-1, dim, dim))
    # unormalized_probs is of shape (#batch_size * #kpts, #row * #cols)
    # where for each batch example all keypoints are serialized before having the next
    # example in the first dimension (exp1_kpt1, exp1_kpt2,.., exp1_kptn, exp2_kpt1, ...)
    unormalized_probs = conv_sum_3D.flatten(2)
    softmax_layer = Softmax(unormalized_probs)

    ########################################
    # defining the parameters of the model #
    ########################################
    # create a list of all model parameters to be fit by gradient descent
    # defininig params of MTFL model
    params = layerSh1.params + layerSh2.params + layerSh3.params + layerFOut.params
    params = params

    #########################
    # measuring the L2 cost #
    #########################
    # L2 norm for common convolutional layers
    L2_sqr = (layerSh1.W ** 2).sum() + (layerSh2.W ** 2).sum() +\
             (layerSh3.W ** 2).sum() + (layerFOut.W **2).sum()
    L2_sqr = L2_sqr

    return conv_sum, softmax_layer, params, L2_sqr


class TCDCN_ConvNet(object):
    def __init__(self, nkerns, rng, decay, conv_size, target_dim=80, **kwargs):
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        dim = target_dim
        num_keypoints = nkerns[0]
        num_img_channels = nkerns[0]
        sys.stderr.write("number of channels is %i\n" %num_img_channels)

        sys.stderr.write("conv_size for structured_model is %i\n" %conv_size)
        #theano.config.compute_test_value = 'raise'
        sys.stderr.write('... building the model\n')
        x = T.tensor4('x', dtype=theano.config.floatX)  # the data is presented in 4D shape (#batch, #kpts, #row, #cols)
        # the given input is the shape (#batch, #row, #cols, #channels)
        # in order to pass it to the first convlolutional layer it should be reshaped to
        # (#batch, #channels, #row, #cols)
        x.tag.test_value = np.random.rand(128, num_keypoints, dim, dim).astype(theano.config.floatX)

        # the keypoint location labels are are presented as a 2D matrix of shape (#batch, #keypoints*2)
        # keypoints are [float32] real values in the range of [0,1]
        y_kpt_norm = T.imatrix('y_kpt_norm')
        y_kpt_norm.tag.test_value = np.random.binomial(n=dim**2, p=0.5, size=(128, num_keypoints)).astype(np.int32)
        # y_kpt_norm_serial is a vector of dim (#batch * #kpts)
        # where for each batch example all keypoints are serialized before having the next
        # example in the first dimension (exp1_kpt1, exp1_kpt2,.., exp1_kptn, exp2_kpt1, ...)
        y_kpt_norm_serial = y_kpt_norm.flatten()
        # y_kpt_ocular_dist is a 1D float vector of (#batch) containing the inter_ocular
        # distance for each training example. It is also a float32 normalized in the range [0,1]
        y_kpt_ocular_dist = T.vector('y_kpt_ocular_dist', dtype=theano.config.floatX)
        y_kpt_ocular_dist.tag.test_value = np.random.rand(128).astype(theano.config.floatX)
        # make a column out of a 1d vector (#batch to #batch x 1)
        y_kpt_ocular = y_kpt_ocular_dist.dimshuffle(0, 'x')

        # the y components for the 300W dataset
        y_kpt_300W = T.matrix('y_kpt_300W', dtype=theano.config.floatX)
        y_kpt_300W.tag.test_value = np.random.rand(128, 10).astype(theano.config.floatX)

        # the labels of the auxiliary tasks are presented as 1D vector of (#batch)
        # [int] labels starting from zero.
        # L2 cost coefficient for the output layer
        L2_coef = T.scalar('L2_coef', dtype=theano.config.floatX)
        L2_coef.tag.test_value = np.float32(1.0)
        # L2 cost coefficient for the fully connected layer

        # mask_kpts is a vector of shape (#batch * #kpts) indicating
        # for each sample which keypoint is on the border.
        # the order of values is: (exp1_kpt1, exp1_kpt2,.., exp1_kptn, exp2_kpt1, ...)
        # mask_kpts is one when the kpt is not in the pad region and zero otherwise.
        mask_kpts = T.vector('mask_kpts', dtype=theano.config.floatX)
        mask_kpts.tag.test_value = (np.ones((128 * num_keypoints))).astype(theano.config.floatX)

        # bound_mask is a matrix of shape (#batch, #kpts) indicating
        # whether for each kpt it is in the padded region or not.
        # bound_mask is one of these values for each kpt:
        # 0: kpt is not in the boundary, 1: kpt is in the left boundary,
        # 2: kpt is in the top boundary
        # 3: kpt is in the right boundary, 4: kpt is in the bottom boundary
        bound_mask = T.matrix('bound_mask', dtype=theano.config.floatX)
        bound_mask.tag.test_value = (np.zeros((128 , num_keypoints))).astype(theano.config.floatX)

        # border_pixel is a matrix of shape (#batch, 4) indicating
        # for each image the starting pixel of valid image (no pad region)
        # it contains a value in the range [0, image_size - 1 ]
        border_pixel = T.matrix('border_pixel', dtype=theano.config.floatX)
        border_pixel.tag.test_value = (np.tile([[0, 0, dim-1 , dim-1]], (128,1))).astype(theano.config.floatX)

        #the input tensor to layer0 is of shape (#batch, #channels, dim * dim)
        layer1_input = x

        dropout = T.scalar('dropout', dtype=theano.config.floatX)
        dropout.tag.test_value = np.float32(0.)

        conv_sum, softmax_layer, params, L2_sqr = fforward_model(layer1_input=layer1_input, dropout=dropout, nkerns=nkerns,
                                                      conv_size=conv_size, num_img_channels=num_img_channels,
                                                      dim=dim, rng=rng)
        self.params = params
        self.L2_sqr = L2_sqr

        ########################################
        # getting the cost of the output layer #
        ########################################
        epsilon = 1e-8
        # cost_batch is a vector of dim (#batch_size * #kpts)
        cost_batch = softmax_layer.negative_log_likelihood(y_kpt_norm_serial)
        cost_kpt = T.sum(cost_batch * mask_kpts) / (T.sum(mask_kpts) + epsilon)
        # cost is the sum of the cost of the keypoints
        cost_kpt *= num_keypoints

        L2_cost = L2_coef * L2_sqr
        cost = L2_cost + cost_kpt

        ########################################################
        # getting the sum log probs for the detected locations #
        ########################################################
        softmax_probs = softmax_layer.p_y_given_x
        max_prob = T.max(softmax_probs, axis=1)
        log_max_prob = -T.log(max_prob)
        # log_max_prob_2D is of shape (#batch, #kpts)
        log_max_prob_2D = log_max_prob.reshape((-1, num_keypoints))
        sum_log_probs = T.sum(log_max_prob_2D, axis=1)

        ##################
        # error for MTFL #
        ##################
        # getting the prediction values
        # predict is of dim (#batch * #kpts)
        predict = softmax_layer.predict(y_kpt_norm_serial)

        # getting the estimated values
        # for each batch, all keypoints come sequentially
        # before seeing the next example.
        # y_pred is of shape (#batch * #kpt)
        y_pred = predict // dim
        # x_pred is of shape (#batch * #kpt)
        x_pred = predict % dim
        # y_true is of shape (#batch * #kpt)
        y_true = y_kpt_norm_serial // dim
        # x_true is of shape (#batch * #kpt)
        x_true = y_kpt_norm_serial % dim

        #################################################
        # getting the model's prediction in [0, dim**2) #
        #################################################
        # predict_2D is of shape (#batch, #kpts)
        predict_2D = predict.reshape((-1, num_keypoints))

        ################################
        # getting error for train set  #
        # masks completely the points  #
        # in the pad rather than       #
        # projecting them to the border#
        ################################
        # getting x_pred and y_pred that is not masked
        # just in training (for illustration purposes
        x_pred_train = x_pred
        y_pred_train = y_pred

        x_diff_sqr_train = (x_pred - x_true)**2
        y_diff_sqr_train = (y_pred - y_true)**2
        # kpt_euc_dist is of shape (#batch * #kpt)
        kpt_euc_dist_train = T.sqrt(x_diff_sqr_train + y_diff_sqr_train)

        # masking the points that are in the pad
        error_kpt_masked_train = kpt_euc_dist_train * mask_kpts

        # error_kpt_2D is of shape (#batch , #kpt)
        error_kpt_2D_train = error_kpt_masked_train.reshape((-1, num_keypoints))

        # the values of x_pred, y_pred, x_true, y_true
        # are in the range of [0,dim). So to make the
        # calculation compatible y_kpt_ocular should also
        # get unnormalized
        y_kpt_ocular_unorm = y_kpt_ocular * dim
        error_kpt_each_norm_MTFL_train = error_kpt_2D_train / y_kpt_ocular_unorm

        # getting the sum of error over all samples and all keypoints
        error_kpt_MTFL_train = T.sum(error_kpt_each_norm_MTFL_train)
        error_kpt_train = error_kpt_MTFL_train

        ##############################
        # getting error for test set #
        # projects the points in the #
        # pad to the border          #
        ##############################
        # moving the kpts to the border if its in the pad #
        # x_pred_2D is of shape (#batch, #kpt)
        x_pred_2D = x_pred.reshape((-1, num_keypoints))
        y_pred_2D = y_pred.reshape((-1, num_keypoints))

        # if bound_mask==1 (left border), use border_pixel[0], elif bound_mask==3 (right border), use border_pixel[2]
        # else, use x_pred_2D
        # x_inside is a matrix of shape (#batch, #kpts) indicating
        x_inside = 1 - (T.eq(bound_mask, 1) + T.eq(bound_mask, 3))
        x_pred_bord = T.eq(bound_mask, 1) * border_pixel[:, 0].reshape((-1, 1)) +\
                      T.eq(bound_mask, 3) * border_pixel[:, 2].reshape((-1, 1)) + x_inside * x_pred_2D
        y_inside = 1 - (T.eq(bound_mask, 2) + T.eq(bound_mask, 4))
        y_pred_bord = T.eq(bound_mask, 2) * border_pixel[:, 1].reshape((-1, 1)) +\
                      T.eq(bound_mask, 4) * border_pixel[:, 3].reshape((-1, 1)) + y_inside * y_pred_2D

        x_inside_all = 1 - (T.lt(x_pred_bord, border_pixel[:, 0].reshape((-1, 1))) +\
                            T.gt(x_pred_bord, border_pixel[:, 2].reshape((-1, 1))))
        x_pred_bord_all = T.lt(x_pred_bord, border_pixel[:, 0].reshape((-1, 1))) * border_pixel[:, 0].reshape((-1, 1)) +\
                          T.gt(x_pred_bord, border_pixel[:, 2].reshape((-1, 1))) * border_pixel[:, 2].reshape((-1, 1)) +\
                          x_inside_all * x_pred_bord
        y_inside_all = 1 - (T.lt(y_pred_bord, border_pixel[:, 1].reshape((-1, 1))) +\
                            T.gt(y_pred_bord, border_pixel[:, 3].reshape((-1, 1))))
        y_pred_bord_all = T.lt(y_pred_bord, border_pixel[:, 1].reshape((-1, 1))) * border_pixel[:, 1].reshape((-1, 1)) +\
                          T.gt(y_pred_bord, border_pixel[:, 3].reshape((-1, 1))) * border_pixel[:, 3].reshape((-1, 1)) +\
                          y_inside_all * y_pred_bord

        # x_pred is of shape (#batch * #kpt)
        x_pred = x_pred_bord_all.flatten()
        y_pred = y_pred_bord_all.flatten()

        x_diff_sqr = (x_pred - x_true) ** 2
        y_diff_sqr = (y_pred - y_true) ** 2
        # kpt_euc_dist is of shape (#batch * #kpt)
        kpt_euc_dist = T.sqrt(x_diff_sqr + y_diff_sqr)

        # applying the mask to the predicted kpts locations
        error_kpt_masked = kpt_euc_dist * mask_kpts

        # error_kpt_2D is of shape (#batch , #kpt)
        error_kpt_2D = error_kpt_masked.reshape((-1, num_keypoints))

        # the values of x_pred, y_pred, x_true, y_true
        # are in the range of [0,dim). So to make the
        # calculation compatible y_kpt_ocular should also
        # get unnormalized
        y_kpt_ocular_unorm = y_kpt_ocular * dim
        error_kpt_each_norm_MTFL = error_kpt_2D / y_kpt_ocular_unorm

        # getting the sum of error over all samples and all keypoints
        error_kpt_MTFL = T.sum(error_kpt_each_norm_MTFL)
        error_kpt = error_kpt_MTFL

        #############################
        # getting errors seperately #
        #############################
        # error_kpt_each contains the error seperately for each batch_sample
        error_kpt_each = T.mean(error_kpt_each_norm_MTFL, axis=1)

        # the same variable for the train set
        error_kpt_each_train = T.mean(error_kpt_each_norm_MTFL_train, axis=1)

        #######################################
        # defining the optimization algorithm #
        #######################################
        # setting the updates using the ada_delta
        self.tr = Train_alg()
        updates = self.tr.build_updates(cost=cost, params=self.params, consider_constant=None, decay=decay)

        ###############################
        # defining the test functions #
        ###############################
        self.train_model = theano.function(
            [L2_coef, x, y_kpt_ocular_dist, y_kpt_norm, mask_kpts, dropout],
            [cost, cost_kpt, L2_cost, error_kpt_train],
            updates=updates, allow_input_downcast=True)

        self.valid_model = theano.function(
            [L2_coef, x, y_kpt_ocular_dist, y_kpt_norm, mask_kpts, dropout],
            [cost, cost_kpt, L2_cost, error_kpt_train], allow_input_downcast=True)

        # testing only on MTFL dataset with no task usage
        self.test_300W = theano.function(
            [L2_coef, x, y_kpt_ocular_dist, y_kpt_norm, bound_mask, mask_kpts, border_pixel, dropout],
            [cost_kpt, L2_cost, error_kpt], allow_input_downcast=True)

        ###########################################
        # Getting the prediction for the test set #
        ###########################################
        x_pred_300W = x_pred.reshape((-1, num_keypoints))
        y_pred_300W = y_pred.reshape((-1, num_keypoints))
        # interleaving x and y values
        kpt_pred = T.zeros_like(x_pred_300W)
        kpt_pred = T.tile(kpt_pred, (1,2))
        kpt_pred = T.set_subtensor(kpt_pred[:,::2], x_pred_300W)
        kpt_pred_300W = T.set_subtensor(kpt_pred[:,1::2], y_pred_300W)
        # kpt_pred_MTFL contains integers in the range [0, dim)

        ############################################
        # Getting the prediction for the train set #
        ############################################
        x_pred_300W_train = x_pred_train.reshape((-1, num_keypoints))
        y_pred_300W_train = y_pred_train.reshape((-1, num_keypoints))
        # interleaving x and y values
        kpt_pred = T.zeros_like(x_pred_300W_train)
        kpt_pred = T.tile(kpt_pred, (1,2))
        kpt_pred = T.set_subtensor(kpt_pred[:,::2], x_pred_300W_train)
        kpt_pred_300W_train = T.set_subtensor(kpt_pred[:,1::2], y_pred_300W_train)

        self.get_keypoints_MTFL = theano.function([x, bound_mask, border_pixel, dropout], kpt_pred_300W, allow_input_downcast=True)
        self.get_keypoints_MTFL_train = theano.function([x, dropout], kpt_pred_300W_train, allow_input_downcast=True)

        self.get_errors = theano.function([x, y_kpt_ocular_dist, y_kpt_norm, bound_mask, border_pixel, mask_kpts, dropout],
                                          error_kpt_each, allow_input_downcast=True)
        self.get_errors_train = theano.function([x, y_kpt_ocular_dist, y_kpt_norm, mask_kpts, dropout],
                                                error_kpt_each_train, allow_input_downcast=True)
        self.model_prediction = theano.function([x, dropout], predict_2D, allow_input_downcast=True)

        self.get_pre_softmax = theano.function([x, dropout], conv_sum, allow_input_downcast=True)
        self.get_sum_log_probs = theano.function([x, dropout], [sum_log_probs, log_max_prob_2D, max_prob], allow_input_downcast=True)

    def dump_params(self, pickle_name):
        "This method dumps the parameters of the TCDCN_ConvNet"
        pickle_param_file = dest_dir + '/' + pickle_name
        with open (pickle_param_file, 'wb') as fp:
            for param in self.params:
                pickle.dump(param.get_value(borrow=True), fp)

    def dump_adadelta_params(self, pickle_name):
        "This method dumps the parameters of adadelta"
        pickle_param_file = dest_dir + '/' + pickle_name
        with open (pickle_param_file, 'wb') as fp:
            for param in self.tr.params:
                pickle.dump(param.get_value(borrow=True), fp)

    def load_params(self, pickle_path, load_adedelta=False):
        # complete pickle path should be given
        "This method loads the parameters of the TCDCN_ConvNet from a pickle file"
        #pickle_param_file = dest_dir + '/' + pickle_name
        pickle_param_file = pickle_path
        with open (pickle_param_file, 'rb') as fp:
            for param in self.params:
               param_vals = pickle.load(fp)
               param.set_value(param_vals)
        print "model params loaded."

        if load_adedelta:
            dirs = pickle_path.split('/')
            if 'epoch' in pickle_path:
                parts = pickle_path.split('epoch_')
                parts = parts[-1].split('_')
                ade_delta_path = '/'.join(dirs[:-1]) + '/adadelta_params_' + '_'.join(parts[1:])
            else:
                parts = pickle_path.split('params')
                ade_delta_path = '/'.join(dirs[:-1]) + '/adadelta_params' + parts[-1]
            with open (ade_delta_path, 'rb') as fp:
                for param in self.tr.params:
                   param_vals = pickle.load(fp)
                   param.set_value(param_vals)
            print "ada_delta params loaded."

def dump_params(pickle_name, params):
    "This method dumps the parameters of the TCDCN_ConvNet"
    pickle_param_file = dest_dir + '/' + pickle_name
    with open (pickle_param_file, 'wb') as fp:
        for param in params:
            pickle.dump(param, fp)

def append_text(file_name, text):
    "This method appends text to file_name in the destination directory"
    file_path = dest_dir + '/' + file_name
    with open (file_path, 'a') as fp:
        fp.write(text)

class Train(object):
    def __init__(self, data_queue, seed_queue, nkerns, num_epochs, batch_size, L2_coef, param_path,
                 file_suffix, sets, param_seed, num_procs, mask_MTFL, mask_300W, nMaps_shuffled,
                 producers, sw_lenght, target_dim, num_queue_elem, param_path_cfNet, decay,
                 train_all_kpts, conv_size, dropout_kpts=False, num_model_kpts=68):

        rng = np.random.RandomState(param_seed)
        sys.stderr.write("rng seed for parameter initialization is %i\n" %param_seed)

        if param_path_cfNet != "":
            #######################################
            # creating coarse_fine convnet model  #
            #######################################
            sys.stderr.write("loading params of coarse_fine_Net from %s \n" %param_path_cfNet)
            tcdcn_cfNet, params_cfNet = create_TCDCN_obejct(param_path_cfNet)
            tcdcn_cfNet.load_params(param_path_cfNet)
            sys.stderr.write("done with loading params \n")
            self.tcdcn_cfNet = tcdcn_cfNet
            self.params_cfNet = params_cfNet
        else:
            sys.stderr.write("using true kpts for training structured_model \n")
            self.tcdcn_cfNet = None

        tcdcn = TCDCN_ConvNet(nkerns=nkerns, rng=rng, decay=decay, conv_size=conv_size, target_dim=target_dim)

        ####################################
        # running a previously saved_model #
        ####################################
        if param_path != "":
            # if param_path is given the params are loaded from file
            sys.stderr.write("loading params from %s \n" %param_path)
            tcdcn.load_params(param_path, load_adedelta=True)

        if '300W' in data_queue.keys():
            self.data_queue_300W = data_queue['300W']
        if 'MTFL' in data_queue.keys():
            self.data_queue_MTFL = data_queue['MTFL']

        self.train_all_kpts = train_all_kpts
        self.seed_queue = seed_queue
        self.num_procs = num_procs
        self.rng = rng
        self.num_queue_elem = num_queue_elem
        self.EOF_used = False
        self.tcdcn = tcdcn
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.L2_coef = L2_coef
        self.file_suffix = file_suffix
        self.sets = sets
        self.mask_MTFL = mask_MTFL
        self.mask_300W = mask_300W
        self.producers = producers
        self.sw_lenght = sw_lenght
        self.dim = target_dim
        self.nMaps_shuffled = nMaps_shuffled
        self.dropout_kpts = dropout_kpts
        self.num_model_kpts = num_model_kpts

    def AddEOF(self):
        if self.mask_MTFL and self.mask_300W:
            if self.SEED_MTFL >= self.max_epochs_MTFL and self.SEED_300W >= self.max_epochs_300W:
                # adding EOF object for as many as the number of processes
                sys.stderr.write("Adding EOF elements to Queue.\n")
                for i in xrange(self.num_procs):
                    self.seed_queue.put(('both', EOF()))
        elif self.mask_MTFL and not self.mask_300W:
            if self.SEED_MTFL >= self.max_epochs_MTFL:
                sys.stderr.write("Adding EOF elements to Queue.\n")
                for i in xrange(self.num_procs):
                    self.seed_queue.put(('MTFL', EOF()))
        elif self.mask_300W and not self.mask_MTFL:
            if self.SEED_300W >= self.max_epochs_300W:
                sys.stderr.write("Adding EOF elements to Queue.\n")
                for i in xrange(self.num_procs):
                    self.seed_queue.put(('300W', EOF()))
        self.EOF_used = True


    def get_mini_batch_train_300W(self):
        # this method gets the next mini-batch only for one or all train datasets.
        index_300W = self.index_300W
        batch_size = self.batch_size
        # getting MTFL mini_batch
        ##############################
        # getting jittered 300W data #
        ##############################
        if self.index_300W == 0:
            seed, next_elem = self.data_queue_300W.get()
            self.train_set_x_300W, set_y = next_elem
            #sys.stderr.write("getting 300W elem for seed %i\n" %seed)

            # getting the mask_border for the train_set
            kpt_norm = set_y['kpt_norm']
            bound_mask = set_y['bound_mask']
            mask_border = get_bound_mask(bound_mask)
            mask_border = mask_padded_kpts(kpt_norm, mask_border)
            set_y['mask_border'] = mask_border

            """
            # getting border_pixels
            pad_ratio = set_y['pad_ratio']
            # change pad_ratio to border pixel locations
            border_pixel = padRatio_to_pixels(pad_ratio, self.train_set_x_300W.shape[1])
            set_y['border_pixel'] = border_pixel
            """

            # making the values discrete
            # getting kpts in the range of [0, dim**2]
            kpt_discret = discretise_y(kpt_norm, self.dim)
            set_y['kpt_norm'] = kpt_discret
            self.train_set_y_300W = set_y

            # putting the seed values
            if self.SEED_300W < self.max_epochs_300W:
                self.seed_queue.put(('300W', self.SEED_300W))
                self.SEED_300W += 1
            else:
                if not self.EOF_used:
                    self.AddEOF()

        x = self.train_set_x_300W[index_300W * batch_size: (index_300W + 1) * batch_size]
        y_ocular_dist = self.train_set_y_300W['ocular_dist'][index_300W * batch_size: (index_300W + 1) * batch_size]
        y_kpt_norm = self.train_set_y_300W['kpt_norm'][index_300W * batch_size: (index_300W + 1) * batch_size]
        # y_mask_border is of shape (#batch, #kpts)
        y_mask_border = self.train_set_y_300W['mask_border'][index_300W * batch_size: (index_300W + 1) * batch_size]

        # updating index values
        self.index_300W = (index_300W + 1) % self.train_nbatches_300W

        # source_points is of shape (#batch, #kpts)
        source_points = get_source_points(self.tcdcn_cfNet, x, y_kpt_norm, self.num_model_kpts)

        # getting the one-hot matrices of the kpt_locations + adding noise to it and
        # getting mask for the noised locations
        # one_hot_maps_4D is of shape (#batch, #kpts, #dim, #dim)
        # mask is of shape (#batch, #kpts)
        one_hot_maps_4D, y_mask_jittered = get_and_noise_one_hot_maps(source_points, self.dim,
                                                                      self.nMaps_shuffled,
                                                                      self.rng, self.dropout_kpts)

        """
        if self.tcdcn_cfNet and self.train_all_kpts:
            # In this case put error on all kpts, except the ones on the border
            mask_kpts = y_mask_border
        else:
            # In this case only put error on the jittered kpts
            # getting the masks where each one value indicates
            # a jittered kpt that is not also on the border.
            mask_kpts = y_mask_border * y_mask_jittered
        """
        mask_kpts = y_mask_border * y_mask_jittered

        #y_mask_kpts is of shape (#batch * #kpts)
        y_mask_kpts = np.ndarray.flatten(mask_kpts)

        # updating the total number of jitted kpts examples
        self.samples_seen += np.sum(y_mask_kpts)

        return [one_hot_maps_4D, y_kpt_norm, y_ocular_dist, y_mask_kpts]


    def get_mini_batch_valid_300W(self, valid_set_x, valid_set_y, index, batch_size):
        # this method gets the next mini-batch only for one datast.
        x = valid_set_x[index * batch_size: (index + 1) * batch_size]
        y_ocular_dist = valid_set_y['ocular_dist'][index * batch_size: (index + 1) * batch_size]
        y_kpt_norm = valid_set_y['kpt_norm'][index * batch_size: (index + 1) * batch_size]
        y_mask_border = valid_set_y['mask_border'][index * batch_size: (index + 1) * batch_size]

        # source_points is of shape (#batch, #kpts)
        source_points = get_source_points(self.tcdcn_cfNet, x, y_kpt_norm, self.num_model_kpts)

        # getting the one-hot matrices of the kpt_locations + adding noise to it and
        # getting mask for the noised locations
        # one_hot_maps_4D is of shape (#batch, #kpts, #dim, #dim)
        # mask is of shape (#batch, #kpts)
        one_hot_maps_4D, y_mask_jittered = get_and_noise_one_hot_maps(y_kpt_norm, self.dim,
                                                                      self.nMaps_shuffled,
                                                                      self.rng, self.dropout_kpts)

        """
        if self.tcdcn_cfNet and self.train_all_kpts:
            # in this case, we are evaluating on the cfNet model's output,
            # so consider all points, if they are not out of border
            mask_kpts = y_mask_border
        else:
            # in this case we are dealing with the true keypoints, so only consider the jittered ones
            # It gets the masks where each one value indicates
            # a jittered kpt that is not also on the border.
            mask_kpts = y_mask_border * y_mask_jittered
        """
        mask_kpts = y_mask_border * y_mask_jittered
        #y_mask_kpts is of shape (#batch * #kpts)
        y_mask_kpts = np.ndarray.flatten(mask_kpts)

        return [one_hot_maps_4D, y_kpt_norm, y_ocular_dist, y_mask_kpts]

    def append_errors(self, error_dict, epoch_sets, epoch, num_samples, is_train=True, batch_sets=None):
        epoch_cost, epoch_cost_kpt, epoch_error_kpt, epoch_l2_cost = epoch_sets
        sw_lenght = self.sw_lenght
        this_epoch_cost = np.mean(epoch_cost)
        error_dict['cost'].append(this_epoch_cost)
        this_epoch_cost_kpt = np.mean(epoch_cost_kpt)
        error_dict['cost_kpt'].append(this_epoch_cost_kpt)
        #epoch_error_kpt = np.sum(np.array(epoch_error_kpt), axis=0)
        #error_dict['error_kpt'].append(epoch_error_kpt/num_samples)
        # summing over all kpts that have been jittered
        # divided by the total number of such examples
        this_epoch_error_kpt_avg = np.sum(epoch_error_kpt)/num_samples
        error_dict['error_kpt_avg'].append(this_epoch_error_kpt_avg)
        error_dict['L2_norm'].append(np.mean(epoch_l2_cost))

        if is_train and batch_sets:
            epoch_cost, epoch_cost_kpt, epoch_l2_cost = batch_sets
            error_dict['cost_batch'].extend(epoch_cost)
            error_dict['cost_kpt_batch'].extend(epoch_cost_kpt)
            error_dict['L2_norm_batch'].extend(epoch_l2_cost)

        if not is_train:
        ##################################################################
        # checking for the best params on keypoint detection cost so far #
        ##################################################################
            if this_epoch_error_kpt_avg < error_dict['min_cost_kpt']:
                error_dict['min_cost_kpt'] = this_epoch_error_kpt_avg
                error_dict['min_cost_kpt_epoch'] = epoch + 1
                error_dict['best_params'] = [param.get_value(borrow=True) for param in self.tcdcn.params]

        if (epoch+1) >= sw_lenght:
            mean_cost_kpt = np.mean(error_dict['cost_kpt'][-sw_lenght:])
            error_dict['cost_kpt_sliding'].append(mean_cost_kpt)
            mean_error_kpt_avg = np.mean(error_dict['error_kpt_avg'][-sw_lenght:])
            error_dict['error_kpt_avg_sliding'].append(mean_error_kpt_avg)
            if mean_cost_kpt < error_dict['min_cost_kpt_sliding']:
                error_dict['min_cost_kpt_sliding'] = mean_cost_kpt
            if mean_error_kpt_avg < error_dict['min_error_kpt_avg_sliding']:
                error_dict['min_error_kpt_avg_sliding'] = mean_error_kpt_avg

        return [this_epoch_cost, this_epoch_cost_kpt]

    def eval_test_set(self, test_set_x, test_set_y, is_MTFL, error_dict, epoch):
        test_num_batches = error_dict['num_batches']
        #test_num_samples = error_dict['num_samples']
        test_num_samples = 0
        sw_lenght = self.sw_lenght
        ##############################
        # getting the test set error #
        ##############################
        batch_size = self.batch_size
        epoch_cost_kpt = []
        epoch_error_kpt = []
        epoch_error_kpt_avg = []
        for index in np.arange(test_num_batches):
            if is_MTFL:
                # adding MTFL data
                x = test_set_x[index * batch_size: (index + 1) * batch_size]
                y_kpt_norm = test_set_y['kpt_norm'][index * batch_size: (index + 1) * batch_size]
                y_kpt_ocular_dist = test_set_y['ocular_dist'][index * batch_size: (index + 1) * batch_size]
                mask_MTFL = np.ones((x.shape[0]))
                # getting the softmax output of the first convnet model
                softmax_output = self.tcdcn_cfNet.get_softmax_output(x, dropout=0)
                cost_kpt, L2_cost, error_kpt, error_kpt_avg = self.tcdcn.test_MTFL(self.L2_coef, hMaps_shuffled,
                                     y_kpt_ocular_dist, y_kpt_norm, mask_MTFL, dropout=0)
            else:
                # adding 300W data
                x = test_set_x[index * batch_size: (index + 1) * batch_size]
                y_kpt_norm = test_set_y['kpt_norm'][index * batch_size: (index + 1) * batch_size]
                y_kpt_ocular_dist = test_set_y['ocular_dist'][index * batch_size: (index + 1) * batch_size]
                y_bound_mask = test_set_y['bound_mask'][index * batch_size: (index + 1) * batch_size]
                y_mask_border = test_set_y['mask_border'][index * batch_size: (index + 1) * batch_size]
                y_border_pixel = test_set_y['border_pixel'][index * batch_size: (index + 1) * batch_size]

                # at test time we consider the one-hot output of
                # the RCN and pass it to the denoising model

                if self.tcdcn_cfNet:
                    one_hot_maps_4D = get_one_hot_predictions(self.tcdcn_cfNet, x, self.dim)
                    # in this case, we are evaluating on the cfNet model's output, so consider all points.
                    mask_kpts = np.ones_like(y_kpt_norm)
                else:
                    source_points = get_source_points(self.tcdcn_cfNet, x, y_kpt_norm, self.num_model_kpts)
                    # getting the one-hot matrices of the kpt_locations + adding noise to it and
                    # getting mask for the noised locations
                    # one_hot_maps_4D is of shape (#batch, #kpts, #dim, #dim)
                    # mask is of shape (#batch, #kpts)
                    one_hot_maps_4D, y_mask_jittered = get_and_noise_one_hot_maps(source_points, self.dim,
                                                                                  self.nMaps_shuffled,
                                                                                  self.rng, self.dropout_kpts)
                    # in this case we are dealing with the true keypoints, so only consider the jittered ones
                    # It gets the masks where each one value indicates
                    # a jittered kpt that is not also on the border.
                    mask_kpts = y_mask_border * y_mask_jittered
                    # Note: the above term gives error on the true keypoint distribution.
                    # if you want to get erro on distance between predicted and sometimes cropped true-values, then use
                    #mask_kpts = y_mask_jittered

                #y_mask_kpts is of shape (#batch * #kpts)
                y_mask_kpts = np.ndarray.flatten(mask_kpts)
                test_num_samples += np.sum(y_mask_kpts)

                cost_kpt, L2_cost, error_kpt = self.tcdcn.test_300W(self.L2_coef, one_hot_maps_4D,
                                                                    y_kpt_ocular_dist, y_kpt_norm,
                                                                    y_bound_mask, y_mask_kpts,
                                                                    y_border_pixel, dropout=0)

            # accumulating the values of the mini-batcjes
            epoch_cost_kpt.append(cost_kpt)
            epoch_error_kpt.append(error_kpt)

        # getting the average of the whole epoch
        avg_epoch_cost_kpt = np.mean(epoch_cost_kpt)
        # getting the average of each keypoint over all of the samples
        #epoch_error_kpt = np.sum(np.array(epoch_error_kpt), axis=0)
        #avg_epoch_error_kpt = epoch_error_kpt/test_num_samples
        avg_epoch_error_kpt_avg = np.sum(epoch_error_kpt)/test_num_samples
        # appending epoch results
        error_dict['cost_kpt'].append(avg_epoch_cost_kpt)
        #error_dict['error_kpt'].append(avg_epoch_error_kpt)
        error_dict['error_kpt_avg'].append(avg_epoch_error_kpt_avg)

        if (epoch+1) >= sw_lenght:
            mean_cost_kpt = np.mean(error_dict['cost_kpt'][-sw_lenght:])
            error_dict['cost_kpt_sliding'].append(mean_cost_kpt)
            mean_error_kpt_avg = np.mean(error_dict['error_kpt_avg'][-sw_lenght:])
            error_dict['error_kpt_avg_sliding'].append(mean_error_kpt_avg)
            if mean_cost_kpt < error_dict['min_cost_kpt_sliding']:
                error_dict['min_cost_kpt_sliding'] = mean_cost_kpt
            if mean_error_kpt_avg < error_dict['min_error_kpt_avg_sliding']:
                error_dict['min_error_kpt_avg_sliding'] = mean_error_kpt_avg

        return avg_epoch_error_kpt_avg

    def train(self):
        # setting the mask for the tasks
        params_pickle_base = 'shared_conv_params'
        self.num_kpts_MTFL = 5
        self.num_kpts_300W = 68

        tcdcn = self.tcdcn
        num_epochs = self.num_epochs
        batch_size = self.batch_size
        L2_coef = self.L2_coef
        file_suffix = self.file_suffix
        mask_MTFL = self.mask_MTFL
        mask_300W = self.mask_300W

        Train, Valid, Test = self.sets
        if mask_MTFL:
            self.train_set_x_MTFL, self.train_set_y_MTFL = Train['MTFL']
            valid_set_x_MTFL, valid_set_y_MTFL = Valid['MTFL']
        elif mask_300W:
            self.train_set_x_300W, self.train_set_y_300W = Train['300W']
            valid_set_x_300W, valid_set_y_300W = Valid['300W']
        else:
            raise ValueError('Neither mask_MTFL=%s nor mask_300W=%s is True' %(mask_MTFL, mask_300W))

        ########################
        # training the convnet #
        ########################
        sys.stderr.write("training starts ...\n")

        start_time = time.time() # start time for training
        period_start_time = time.time() # start time for the saving model
        save_tresh_mins = 360. # setting the time threshold for saving the model params to four hours
        epoch_100_start_time = time.time() # start time for training

        # since the datasets are trained simultaneously, The results are kept in one OrderedDict
        Train_error = OrderedDict()
        Train_error['cost'] = []                    # total cost for training (keypoint cost + L2 coef + other if applicable)
        Train_error['cost_kpt'] = []                # keypoint cost for training
        Train_error['cost_kpt_sliding'] = []        # keypoint cost for training, measured by taking a sliding window of train_total_cost_kpt
        Train_error['cost_gl'] = []
        Train_error['cost_gen'] = []
        Train_error['cost_sm'] = []
        Train_error['cost_pose'] = []
        Train_error['error_test'] = []              # total error for the auxiliary tasks
        Train_error['error_kpt'] = []               # individual error for each keypoint location (the test time evaluation metric)
        Train_error['error_kpt_avg'] = []           # average of the error for all keypoints
        Train_error['error_kpt_avg_sliding'] = []   # average of the error for all keypoints, measured by taking a sliding windows of train_total_error_kpt_avg
        Train_error['L2_norm'] = []
        Train_error['min_cost_kpt_sliding'] = np.inf
        Train_error['min_error_kpt_avg_sliding'] = np.inf
        # batch data
        Train_error['cost_batch'] = []
        Train_error['cost_kpt_batch'] = []
        Train_error['L2_norm_batch'] = []
        Train_error['cost_gl_batch'] = []
        Train_error['cost_gen_batch'] = []
        Train_error['cost_sm_batch'] = []
        Train_error['cost_pose_batch'] = []

        # the results are kept separately for each valid set
        Valid_error = OrderedDict()
        for subset in Valid.keys():
            setx, sety = Valid[subset]
            subset_dict = OrderedDict()
            subset_dict['num_batches'] = int(np.ceil(setx.shape[0]/float(batch_size)))
            subset_dict['num_samples'] = setx.shape[0]
            subset_dict['cost'] = []                    # total cost for training (keypoint cost + L2 coef + other if applicable)
            subset_dict['cost_kpt'] = []                # keypoint cost for training
            subset_dict['cost_kpt_sliding'] = []        # keypoint cost for training, measured by taking a sliding window of train_total_cost_kpt
            subset_dict['cost_gl'] = []
            subset_dict['cost_gen'] = []
            subset_dict['cost_sm'] = []
            subset_dict['cost_pose'] = []
            subset_dict['error_test'] = []              # total error for the auxiliary tasks
            subset_dict['error_kpt'] = []               # individual error for each keypoint location (the test time evaluation metric)
            subset_dict['error_kpt_avg'] = []           # average of the error for all keypoints
            subset_dict['error_kpt_avg_sliding'] = []   # average of the error for all keypoints, measured by taking a sliding windows of train_total_error_kpt_avg
            subset_dict['L2_norm'] = []
            subset_dict['min_cost_kpt_sliding'] = np.inf
            subset_dict['min_error_kpt_avg_sliding'] = np.inf
            subset_dict['min_cost_kpt'] = np.inf
            subset_dict['min_cost_kpt_epoch'] = -1
            subset_dict['best_params'] = []             # the parameters of the best model for this validation set
            Valid_error[subset] = subset_dict
            ###########################
            # getting the mask_border #
            ###########################
            kpt_norm = sety['kpt_norm']
            bound_mask = sety['bound_mask']
            mask_border = get_bound_mask(bound_mask)
            mask_border = mask_padded_kpts(kpt_norm, mask_border)
            sety['mask_border'] = mask_border
            #########################
            # getting border_pixels #
            #########################
            pad_ratio = sety['pad_ratio']
            border_pixel = padRatio_to_pixels(pad_ratio, setx.shape[1])
            sety['border_pixel'] = border_pixel
            ##############################
            # making the values discrete #
            ##############################
            # getting kpts in the range of [0, dim**2]
            kpt_discret = discretise_y(kpt_norm, self.dim)
            sety['kpt_norm'] = kpt_discret
            Valid[subset] = (setx, sety)

        Test_error = OrderedDict()
        for dset in Test.keys():
            dset_dict = OrderedDict()
            for subset in Test[dset].keys():
                setx, sety = Test[dset][subset]
                subset_dict = OrderedDict()
                subset_dict['num_batches'] = int(np.ceil(setx.shape[0]/float(batch_size)))
                subset_dict['num_samples'] = setx.shape[0]
                subset_dict['cost_kpt'] = []
                subset_dict['cost_kpt_sliding'] = []
                subset_dict['error_kpt'] = []
                subset_dict['error_kpt_avg'] = []
                subset_dict['error_kpt_avg_sliding'] = []
                subset_dict['min_cost_kpt_sliding'] = np.inf
                subset_dict['min_error_kpt_avg_sliding'] = np.inf
                dset_dict[subset] = subset_dict
                ###########################
                # getting the mask_border #
                ###########################
                kpt_norm = sety['kpt_norm']
                bound_mask = sety['bound_mask']
                mask_border = get_bound_mask(bound_mask)
                mask_border = mask_padded_kpts(kpt_norm, mask_border)
                sety['mask_border'] = mask_border
                #########################
                # getting border_pixels #
                #########################
                pad_ratio = sety['pad_ratio']
                border_pixel = padRatio_to_pixels(pad_ratio, setx.shape[1])
                sety['border_pixel'] = border_pixel
                ##############################
                # making the values discrete #
                ##############################
                # getting kpts in the range of [0, dim**2]
                kpt_discret = discretise_y(kpt_norm, self.dim)
                sety['kpt_norm'] = kpt_discret
                Test[dset][subset] = (setx, sety)
            Test_error[dset] = dset_dict

        '''
        # dumping the params before start of the model
        params_pickle_name = params_pickle_base + '_epoch_0' + file_suffix + '.pickle'
        tcdcn.dump_params(params_pickle_name)

        # testing the dumped values by previously trained model"
        params_pickle_name = params_pickle_base + '_epoch_' + str(num_epochs) + '.pickle'
        tcdcn.load_params(params_pickle_name)
        print ' printing the values'
        params = tcdcn.get_params()
        for param in params:
            print "params %s" %(param,)
        '''

        ########################
        # dataset based values #
        ########################
        # the index for minibatches while trainning
        if Train.has_key('MTFL'):
            setx, sety = Train['MTFL']
            self.train_nbatches_MTFL = int(np.ceil(setx.shape[0]/float(batch_size)))
        if Train.has_key('300W'):
            setx, sety = Train['300W']
            self.train_nbatches_300W = int(np.ceil(setx.shape[0]/float(batch_size)))
        self.index_MTFL = 0
        self.index_300W = 0
        # initialing seed value
        self.SEED_MTFL = self.num_queue_elem
        self.SEED_300W = self.num_queue_elem
        # specifying the number of updates in an epoch
        if mask_MTFL and mask_300W:
            per_epoch_updates = 80 #number of updates in an epoch
        elif mask_MTFL:
            per_epoch_updates = self.train_nbatches_MTFL
        elif mask_300W:
            per_epoch_updates = self.train_nbatches_300W
        else:
            raise Exception('none of the masks is True')

        self.total_updates = per_epoch_updates * num_epochs # total number of updates in training
        if mask_MTFL:
            self.max_epochs_MTFL = np.ceil(self.total_updates / float(self.train_nbatches_MTFL))
        if mask_300W:
            self.max_epochs_300W = np.ceil(self.total_updates / float(self.train_nbatches_300W))

        #################################
        # going through training epochs #
        #################################
        # running the thread for training the model
        # each iteratiob of this while loop is one iteration of epoch
        #epoch = -1
        #while True:
        sys.stderr.write("Starting the first epoch.\n")
        for epoch in xrange(num_epochs):
            # checking whether child processes are stil alive
            if self.producers:
                for pr in self.producers:
                    if pr.exitcode > 0:
                        sys.stderr.write("An error encountered in one of the child processes. exiting ...%i\n")
                        exit()
            #sys.stderr.write("training epoch %i\n" %(epoch+1))
            epoch_start_time = time.time()
            epoch_cost = []
            epoch_error_test = []
            epoch_cost_kpt = []
            epoch_error_kpt = []
            epoch_cost_gl = []
            epoch_cost_gen = []
            epoch_cost_sm = []
            epoch_cost_pose = []
            epoch_lambda_gl = []
            epoch_lambda_gen = []
            epoch_lambda_sm = []
            epoch_lambda_pose = []
            epoch_l2_cost = []

            # the number of seen examples in this epoch
            self.samples_seen = 0

            for upd in xrange(per_epoch_updates):
                one_hot_maps_4D, y_kpt_norm, y_kpt_ocular_dist, y_mask_kpts = self.get_mini_batch_train_300W()
                # getting values in the range of [0, dim**2]
                if epoch < 0: # in the first epoch, we just evaluate the performance of random initialization without any parameter update
                    # note that since the model is not trained in the first epoch, the valid and test sets cost and errors for the first epoch
                    # would be the model's performace before training
                    cost, cost_kpt, L2_cost, error_kpt = tcdcn.valid_model(L2_coef, one_hot_maps_4D, y_kpt_ocular_dist, y_kpt_norm,
                                                                           y_mask_kpts, dropout=0)
                else:
                    #batch_start = time.time() # the end of training time
                    cost, cost_kpt, L2_cost, error_kpt = tcdcn.train_model(L2_coef, one_hot_maps_4D, y_kpt_ocular_dist, y_kpt_norm,
                                                                           y_mask_kpts, dropout=0)
                    #batch_end = time.time() # the end of training time
                    #batch_time = (batch_end - batch_start)
                    #sys.stderr.write('Training took %f minutes\n' % (batch_time / 60.))

                epoch_cost.append(cost)
                #epoch_error_test.append(error_test)
                epoch_cost_kpt.append(cost_kpt)
                epoch_l2_cost.append(L2_cost)
                epoch_error_kpt.append(error_kpt)

            # appending epoch results
            epoch_sets = [epoch_cost, epoch_cost_kpt, epoch_error_kpt, epoch_l2_cost]
            #######################################################
            # saving mini-batch logs only for the first 50 epochs #
            #######################################################
            if epoch < 20:
                #batch_sets = [epoch_cost, epoch_cost_kpt, epoch_l2_cost, epoch_cost_gl, epoch_cost_gen, epoch_cost_sm, epoch_cost_pose]
                batch_sets = [epoch_cost, epoch_cost_kpt, epoch_l2_cost]
            else:
                batch_sets = None
            num_samples = self.samples_seen
            train_epoch_cost, train_epoch_cost_kpt = self.append_errors(error_dict=Train_error, epoch_sets=epoch_sets, epoch=epoch,
                          num_samples=num_samples, is_train=True, batch_sets=batch_sets)

            ################################
            # getting the valid set error #
            ###############################
            for subset in Valid.keys():
                setx, sety = Valid[subset]
                valid_error = Valid_error[subset]
                num_batches = valid_error['num_batches']
                num_samples = 0
                batch_size = self.batch_size
                epoch_cost = []
                epoch_error_test = []
                epoch_cost_kpt = []
                epoch_error_kpt = []
                epoch_cost_gl = []
                epoch_cost_gen = []
                epoch_cost_sm = []
                epoch_cost_pose = []
                epoch_l2_cost = []
                for index in np.arange(num_batches):
                    # getting the next mini-batch for the validation set
                    one_hot_maps_4D, y_kpt_norm, y_kpt_ocular_dist, y_mask_kpts =\
                        self.get_mini_batch_valid_300W(setx, sety, index, batch_size)
                    # evaluating performance on the validation batch
                    cost, cost_kpt, L2_cost, error_kpt = tcdcn.valid_model(L2_coef, one_hot_maps_4D, y_kpt_ocular_dist, y_kpt_norm,
                                                                           y_mask_kpts, dropout=0)
                    #[cost_gl, cost_gen, cost_sm, cost_pose] = task_cost_vec
                    epoch_cost.append(cost)
                    #epoch_error_test.append(error_test)
                    epoch_cost_kpt.append(cost_kpt)
                    epoch_error_kpt.append(error_kpt)
                    epoch_l2_cost.append(L2_cost)
                    num_samples += np.sum(y_mask_kpts)

                # appending epoch results
                epoch_sets = [epoch_cost, epoch_cost_kpt, epoch_error_kpt, epoch_l2_cost]
                batch_sets = None
                #num_samples = Valid_error[subset]['num_samples']
                valid_epoch_cost, valid_epoch_cost_kpt = self.append_errors(error_dict=Valid_error[subset], epoch_sets=epoch_sets, epoch=epoch,\
                              num_samples=num_samples, is_train=False, batch_sets=batch_sets)

            ##############################
            # getting the test set error #
            ##############################
            test_epoch_error_kpt_avg = []
            for dset in Test.keys():
                for subset in Test[dset].keys():
                    setx, sety = Test[dset][subset]
                    error_dict = Test_error[dset][subset]
                    if dset == 'MTFL':
                        is_MTFL=True
                    else:
                        is_MTFL=False

                    epoch_err_kpt_avg = self.eval_test_set(test_set_x=setx, test_set_y=sety, is_MTFL=is_MTFL, error_dict=error_dict, epoch=epoch)
                    name = "%s_%s" %(dset, subset)
                    test_epoch_error_kpt_avg.append([name, epoch_err_kpt_avg])

            if epoch == 1 :
                 sys.stderr.write("done with the first epochs of valid and train\n")
            #print "done with the epoch %i" %(epoch + 1)

            ####################################################################
            # dumping the params of the best model after a fixed time-interval #
            ####################################################################
            current_time = time.time()
            if ((current_time - period_start_time)/ 60.) >= save_tresh_mins:
                # saving the best model for each valid set
                for subset in Valid_error.keys():
                    valid_set = Valid_error[subset]
                    best_params = valid_set['best_params']
                    params_pickle_name = params_pickle_base + file_suffix + '_' + subset + '.pickle'
                    dump_params(params_pickle_name, best_params)

                    # dumping the adadelta params at the end of training for the last epoch
                    params_pickle_name = 'adadelta_params' + file_suffix + '.pickle'
                    tcdcn.dump_adadelta_params(params_pickle_name)

                # writing the epoch number to a txt file
                text_file_name = params_pickle_base + file_suffix + '.txt'
                text = "saved model params in epoch %i\n" %epoch
                append_text(text_file_name, text)
                period_start_time = current_time

            num_epoch_msg = 1
            if (epoch+1) % num_epoch_msg == 0 :
                sys.stderr.write("done with epoch %i\n" %(epoch + 1))
                epoch_100_end_time = time.time() # start time for training
                epoch_training_time = (epoch_100_end_time - epoch_100_start_time)
                sys.stderr.write('%i epochs took %f minutes\n' % (num_epoch_msg, epoch_training_time / 60.))
                epoch_100_start_time = epoch_100_end_time
                sys.stderr.write('train epoch cost is %s\n' %(train_epoch_cost))
                sys.stderr.write('train epoch cost_kpt is %s\n' %(train_epoch_cost_kpt))
                sys.stderr.write('valid epoch cost is %s\n' %(valid_epoch_cost))
                sys.stderr.write('valid epoch cost_kpt is %s\n' %(valid_epoch_cost_kpt))
                sys.stderr.write('test epoch error_kpt_avg is %s\n' %(test_epoch_error_kpt_avg,))

        ##########################################
        # outside the loop of training the model #
        ##########################################
        end_time = time.time() # the end of training time
        training_time = (end_time - start_time)
        sys.stderr.write('Training took %f minutes\n' % (training_time / 60.))

        ################################################
        # saving the costs and errors and model params #
        ################################################
        # dumping the params at the end of training for the last epoch
        sys.stderr.write('Saving Model.\n')
        params_pickle_name = params_pickle_base + '_epoch_' + str(num_epochs) + file_suffix + '.pickle'
        tcdcn.dump_params(params_pickle_name)

        # dumping the params of the best model for each valid set, if it is not the last model
        for subset in Valid_error.keys():
            valid_set = Valid_error[subset]
            best_params = valid_set['best_params']
            min_cost_kpt_epoch = valid_set['min_cost_kpt_epoch']
            params_pickle_name = params_pickle_base + file_suffix + '_' + subset + '.pickle'
            dump_params(params_pickle_name, best_params)

        # dumping the adadelta params at the end of training for the last epoch
        params_pickle_name = 'adadelta_params' + file_suffix + '.pickle'
        tcdcn.dump_adadelta_params(params_pickle_name)

        '''
        print ' printing the values'
        params = tcdcn.get_params()
        for param in params:
            print "params %s" %(param,)
        '''

        # saving the error and the cost #
        # 'train', 'valid' and 'test' sets have the following components.
        #
        # 'cost' : Eq. 3 of the TCDCN paper, which is the regression error for the keypoints +
        # negative log likelihood for the 4 regression tasks + the L2-norm of the weights
        #
        # 'error_test' : the sum of the error for the 4 regression tasks. for each regression task
        # the error at test time, is a float value representing the number of errors in the set (train or test)
        # over the total number of examples of the set. Note that the error is returned for all tasks regardless of
        # whether the task was used in the cost during training or not.
        #
        # 'cost_kpt' : the cost only for the keypoints
        #
        # 'error_kpt' : the average of (sqrt((kpt_x_out - kpt_x_true)^2 + (kpt_y_out - kpt_y_true)^2)
        # for each keypoint) over the dataset normalized by the inter-ocular distance.
        # error_kpt is a vector of shape (#keypoints)
        #
        # 'error_kpt_avg' : the average of the above value, which is the average over all keypoints.
        #
        # 'cost_gl' to 'cost_pose': the cost of the logisic regression of the layer estimating one of wearing_glasses, gender, smiling, pose.
        #  Note that the error is returned for all tasks regardless of whether the task was used in the cost during training or not.
        #
        # 'L2_norm' : the L2_norm cost during training
        #
        # Note: the features ending in '_batch' are quivalent to the above ones with the exception that they are gathered while training the model on the train_set
        # for each mini-batch, which is different from other (non batch) values that are the average for each epoch.

        # orderedDict for all sets
        error = OrderedDict()
        error['train'] = Train_error
        error['valid'] = Valid_error
        error['test'] = Test_error

        # saving the error values in a pickle file
        sys.stderr.write('Saving Logs.\n')
        error_file = dest_dir + '/epochs_log' + file_suffix + '.pickle'
        with open(error_file, 'wb') as fp:
            pickle.dump(error, fp)

        sys.stderr.write('Getting Min Errors.\n')
        message = "min train cost_kpt:%f\n" %Train_error['min_cost_kpt_sliding']
        message += "min train error_kpt-avg:%f\n" %Train_error['min_error_kpt_avg_sliding']

        for subset in Valid_error.keys():
            message += "min valid %s cost_kpt:%f\n" %(subset, Valid_error[subset]['min_cost_kpt_sliding'])
            message += "min valid %s error_kpt_avg:%f\n" %(subset, Valid_error[subset]['min_error_kpt_avg_sliding'])

        for dset in Test_error.keys():
            dset_errors = Test_error[dset]
            for subset in dset_errors.keys():
                message += "min test %s cost_kpt:%f\n" %(subset, dset_errors[subset]['min_cost_kpt_sliding'])
                message += "min test %s error_kpt_avg:%f\n" %(subset, dset_errors[subset]['min_error_kpt_avg_sliding'])

        return message
