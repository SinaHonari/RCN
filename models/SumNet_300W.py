import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from collections import OrderedDict
import cPickle as pickle
import time
import sys
sys.setrecursionlimit(2000)
import RCN
from RCN.utils.grad_updates import Train_alg
from RCN.preprocessing.tools import (EOF, padRatio_to_pixels, limit_x,
                                     discretise_y, mask_padded_kpts, get_bound_mask)
from RCN.utils.bilinear import bilinear_weights
from RCN.preprocessing.local_contrast_normalization import lcn
from RCN.models.layers import (LogisticRegression, ConvPoolLayer,
                               HiddenLayer, Softmax, PoolLayer)
import os

source_dir = os.path.dirname(RCN.__file__)
dest_dir = source_dir + '/models/exp_shared_conv'


def fforward_model(layer1_input, dropout, nkerns, num_img_channels, dim, rng,
                   use_res_2=False, bilinear=False, weight_per_pixel=False,
                   extra_fine=False, bch_norm=False, coarse_conv_size=3,
                   conv_per_kpt=False, linear_conv_per_kpt=False, **kwargs):

    ############################
    # building the conv layers #
    ############################
    num_keypoints = nkerns[0]
    # the number of branches in the model
    num_branch = 0
    conv_large = coarse_conv_size
    conv_border = conv_large / 2
    if coarse_conv_size == 5:
        conv_small = 1
    elif coarse_conv_size == 3:
        conv_small = 3

    if conv_per_kpt:
        act_final_branch = 'relu'
    else:
        act_final_branch = 'linear'

    #####################
    ### shared layer1 ###
    #####################
    layerSh1 = ConvPoolLayer(
        rng,
        dropout = dropout,
        input=layer1_input,
        image_shape=(None, num_img_channels, None, None),
        filter_shape=(nkerns[1], num_img_channels, 3, 3),
        act='relu',
        pool=False,
        border_mode='full'
    )
    layerSh1_output = layerSh1.output[:, :, 1:-1, 1:-1]
    # layerSh1_output is now of shape (#batch, #nkerns[1]=16, 80, 80)
    layerSh1.W.name = 'Conv_layerSh1_W'
    layerSh1.b.name = 'Conv_layerSh1_b'
    layerSh1 = layerSh1

    ########################
    # Fine branch layers F #
    ########################
    num_branch += 1

    layerF1 = ConvPoolLayer(
        rng,
        dropout = dropout,
        input=layerSh1_output,
        image_shape=(None, nkerns[1], None, None),
        filter_shape=(nkerns[2], nkerns[1], conv_large, conv_large),
        act='relu',
        pool=False,
        border_mode='full'
    )
    layerF1.W.name = 'Conv_layerF1_W'
    layerF1.b.name = 'Conv_layerF1_b'
    layerF1 = layerF1
    layerF1_output = layerF1.output[:, :, conv_border:-conv_border, conv_border:-conv_border]
    # layerF1_output is now of shape (#batch, #nkerns[2]=16, 80, 80)

    layerF2 = ConvPoolLayer(
        rng,
        dropout = dropout,
        input=layerF1_output,
        image_shape=(None, nkerns[2], None, None),
        filter_shape=(nkerns[2], nkerns[2], conv_small, conv_small),
        act=act_final_branch,
        pool=False,
        border_mode='full'
    )
    layerF2.W.name = 'Conv_layerF2_W'
    layerF2.b.name = 'Conv_layerF2_b'
    layerF2 = layerF2
    if conv_small != 1:
        layerF2_output = layerF2.output[:, :, conv_border:-conv_border, conv_border:-conv_border]
    else:
        layerF2_output = layerF2.output
    # layerF2_output is now of shape (#batch, #nkerns[0]=5, 80, 80)

    # third F layer
    layerF3 = ConvPoolLayer(
        rng,
        dropout = dropout,
        bch_norm = bch_norm,
        input=layerF2_output,
        image_shape=(None, nkerns[2], None, None),
        filter_shape=(nkerns[2], nkerns[2], conv_small, conv_small),
        act='relu',
        pool=False,
        border_mode='full'
    )
    layerF3.W.name = 'Conv_layerF3_W'
    layerF3.b.name = 'Conv_layerF3_b'
    if conv_small != 1:
        layerF3_output = layerF3.output[:, :, conv_border:-conv_border, conv_border:-conv_border]
    else:
        layerF3_output = layerF3.output
    # layerF3_output is now of shape (#batch, #nkerns[0]=5, 80, 80)

    # specifying the input the layerFOut
    layer_FOut_input = layerF3_output

    if extra_fine:
        # forth F layer
        layerF4 = ConvPoolLayer(
            rng,
            dropout = dropout,
            bch_norm = bch_norm,
            input=layerF3_output,
            image_shape=(None, nkerns[2], None, None),
            filter_shape=(nkerns[2], nkerns[2], conv_small, conv_small),
            act='relu',
            pool=False,
            border_mode='full'
        )
        layerF4.W.name = 'Conv_layerF4_W'
        layerF4.b.name = 'Conv_layerF4_b'
        if conv_small != 1:
            layerF4_output = layerF4.output[:, :, conv_border:-conv_border, conv_border:-conv_border]
        else:
            layerF4_output = layerF4.output

        # fifth F layer
        layerF5 = ConvPoolLayer(
            rng,
            dropout = dropout,
            bch_norm = bch_norm,
            input=layerF4_output,
            image_shape=(None, nkerns[2], None, None),
            filter_shape=(nkerns[2], nkerns[2], conv_small, conv_small),
            act='relu',
            pool=False,
            border_mode='full'
        )
        layerF5.W.name = 'Conv_layerF5_W'
        layerF5.b.name = 'Conv_layerF5_b'
        if conv_small != 1:
            layerF5_output = layerF5.output[:, :, conv_border:-conv_border, conv_border:-conv_border]
        else:
            layerF5_output = layerF5.output

        # specifying the input layer to layerFOut
        layer_FOut_input = layerF5_output

    # Output F layer
    layerFOut = ConvPoolLayer(
        rng,
        dropout = dropout,
        bch_norm = bch_norm,
        input=layer_FOut_input,
        image_shape=(None, nkerns[2], None, None),
        filter_shape=(nkerns[0], nkerns[2], conv_small, conv_small),
        act='linear',
        pool=False,
        border_mode='full'
    )
    layerFOut.W.name = 'Conv_layerFOut_W'
    layerFOut.b.name = 'Conv_layerFOut_b'
    if conv_small != 1:
        layerFOut_output = layerFOut.output[:, :, conv_border:-conv_border, conv_border:-conv_border]
    else:
        layerFOut_output = layerFOut.output
    # layerFOut_output is now of shape (#batch, #nkerns[0]=5, 80, 80)

    ######################
    ### shared layers2 ###
    ######################
    # shared pool layer
    layerP1 = PoolLayer(
        input=layerSh1_output,
        dropout = dropout,
        poolsize=(2,2),
        ignore_border=False
    )
    layerP1_output = layerP1.output
    # layerP1_output is now of shape (#batch, #nkerns[1]=16, 40, 40)
    layerP1 = layerP1

    # shared conv layer
    layerSh2 = ConvPoolLayer(
        rng,
        dropout = dropout,
        input=layerP1_output,
        image_shape=(None, nkerns[1], None, None),
        filter_shape=(nkerns[3], nkerns[1], 3, 3),
        act='relu',
        pool=False,
        border_mode='full'
    )
    layerSh2_output = layerSh2.output[:, :, 1:-1, 1:-1]
    # layerSh2_output is now of shape (#batch, #nkerns[3]=32, 40, 40)
    layerSh2.W.name = 'Conv_layerSh2_W'
    layerSh2.b.name = 'Conv_layerSh2_b'
    layerSh2 = layerSh2

    ##########################
    # Middle branch layers M #
    ##########################
    num_branch += 1

    # first conv layer
    layerM1 = ConvPoolLayer(
        rng,
        dropout = dropout,
        input=layerSh2_output,
        image_shape=(None, nkerns[3], None, None),
        filter_shape=(nkerns[4], nkerns[3], conv_large, conv_large),
        act='relu',
        pool=False,
        border_mode='full'
    )
    layerM1.W.name = 'Conv_layerM1_W'
    layerM1.b.name = 'Conv_layerM1_b'
    layerM1 = layerM1
    layerM1_output = layerM1.output[:, :, conv_border:-conv_border, conv_border:-conv_border]
    # layerM1_output is now of shape (#batch, #nkerns[4]=32, 40, 40)

    # second conv layer
    layerM2 = ConvPoolLayer(
        rng,
        dropout = dropout,
        input=layerM1_output,
        image_shape=(None, nkerns[4], None, None),
        filter_shape=(nkerns[0], nkerns[4], conv_small, conv_small),
        act=act_final_branch,
        pool=False,
        border_mode='full'
    )
    layerM2.W.name = 'Conv_layerM2_W'
    layerM2.b.name = 'Conv_layerM2_b'
    layerM2 = layerM2
    if conv_small != 1:
        layerM2_output = layerM2.output[:, :, conv_border:-conv_border, conv_border:-conv_border]
    else:
        layerM2_output = layerM2.output
    # layerM2_output is now of shape (#batch, #nkerns[0]=5, 40, 40)

    ##########################
    # upsampling layer for M #
    ##########################
    ratio = 2
    if bilinear:
        zero_tensor = T.zeros_like(layerF2_output)
        # copying the values of the coarse model equally spaces to a zero matrix
        layerM3_input = T.set_subtensor(zero_tensor[:, :, ::ratio, ::ratio], layerM2_output)
        layerM3_filters = bilinear_weights(dim=nkerns[0], ratio=ratio).astype(theano.config.floatX)
        index_start = layerM3_filters.shape[-1]/2

        layerM3 = conv.conv2d(
            input=layerM3_input,
            filters=layerM3_filters,
            image_shape=(None, nkerns[0], None, None),
            filter_shape=layerM3_filters.shape,
            border_mode='full'
        )
        layerM3 = layerM3
        layerM3_output = layerM3[:, :, index_start: -index_start, index_start: -index_start]
        # layerM3_output is now of shape (#batch, #nkerns[0]=5, 80, 80)
    else:
        layerM3_upsample = T.extra_ops.repeat(layerM2_output, repeats=ratio, axis=-1)
        layerM3_output = T.extra_ops.repeat(layerM3_upsample, repeats=ratio, axis=-2)

    ######################
    ### shared layers3 ###
    ######################
    # shared pool layer
    layerP2 = PoolLayer(
        input=layerSh2_output,
        dropout = dropout,
        poolsize=(2,2),
        ignore_border=False
    )
    layerP2_output = layerP2.output
    # layerP2_output is now of shape (#batch, #nkerns[3]=32, 20, 20)
    layerP2 = layerP2

    # shared conv layer
    layerSh3 = ConvPoolLayer(
        rng,
        dropout = dropout,
        input=layerP2_output,
        image_shape=(None, nkerns[3], None, None),
        filter_shape=(nkerns[5], nkerns[3], 3, 3),
        act='relu',
        pool=False,
        border_mode='full'
    )
    layerSh3_output = layerSh3.output[:, :, 1:-1, 1:-1]
    # layerSh3_output is now of shape (#batch, #nkerns[5]=48, 20, 20)
    layerSh3.W.name = 'Conv_layerSh3_W'
    layerSh3.b.name = 'Conv_layerSh3_b'
    layerSh3 = layerSh3

    ##########################
    # Coarse branch layers C #
    ##########################
    num_branch += 1

    # first conv layer
    layerC1 = ConvPoolLayer(
        rng,
        dropout = dropout,
        input=layerSh3_output,
        image_shape=(None, nkerns[5], None, None),
        filter_shape=(nkerns[6], nkerns[5], conv_large, conv_large),
        act='relu',
        pool=False,
        border_mode='full'
    )
    layerC1.W.name = 'Conv_layerC1_W'
    layerC1.b.name = 'Conv_layerC1_b'
    layerC1 = layerC1
    layerC1_output = layerC1.output[:, :, conv_border:-conv_border, conv_border:-conv_border]
    # layerC1_output is now of shape (#batch, #nkerns[6]=48, 20, 20)

    # second conv layer
    layerC2 = ConvPoolLayer(
        rng,
        dropout = dropout,
        input=layerC1_output,
        image_shape=(None, nkerns[6], None, None),
        filter_shape=(nkerns[0], nkerns[6], conv_small, conv_small),
        act=act_final_branch,
        pool=False,
        border_mode='full'
    )
    layerC2.W.name = 'Conv_layerC2_W'
    layerC2.b.name = 'Conv_layerC2_b'
    layerC2 = layerC2
    if conv_small != 1:
        layerC2_output = layerC2.output[:, :, conv_border:-conv_border, conv_border:-conv_border]
    else:
        layerC2_output = layerC2.output
    # layerC2_output is now of shape (#batch, #nkerns[0]=5, 20, 20)

    ##########################
    # upsampling layer for C #
    ##########################
    ratio = 4
    if bilinear:
        zero_tensor = T.zeros_like(layerF2_output)
        # copying the values of the coarse model equally spaces to a zero matrix
        layerC3_input = T.set_subtensor(zero_tensor[:, :, ::ratio, ::ratio], layerC2_output)
        layerC3_filters = bilinear_weights(dim=nkerns[0], ratio=ratio).astype(theano.config.floatX)
        index_start = layerC3_filters.shape[-1]/2

        layerC3 = conv.conv2d(
            input=layerC3_input,
            filters=layerC3_filters,
            image_shape=(None, nkerns[0], None, None),
            filter_shape=layerC3_filters.shape,
            border_mode='full'
        )
        layerC3 = layerC3
        layerC3_output = layerC3[:, :, index_start: -index_start, index_start: -index_start]
        # layerC3_output is now of shape (#batch, #nkerns[0]=5, 80, 80)
    else:
        layerC3_upsample = T.extra_ops.repeat(layerC2_output, repeats=ratio, axis=-1)
        layerC3_output = T.extra_ops.repeat(layerC3_upsample, repeats=ratio, axis=-2)

    ######################
    ### shared layers4 ###
    ######################
    # shared pool layer
    layerP3 = PoolLayer(
        input=layerSh3_output,
        dropout = dropout,
        poolsize=(2,2),
        ignore_border=False
    )
    layerP3_output = layerP3.output
    # layerP3_output is now of shape (#batch, #nkerns[5]=48, 10, 10)
    layerP3 = layerP3

    # shared conv layer
    layerSh4 = ConvPoolLayer(
        rng,
        dropout = dropout,
        input=layerP3_output,
        image_shape=(None, nkerns[5], None, None),
        filter_shape=(nkerns[7], nkerns[5], 3, 3),
        act='relu',
        pool=False,
        border_mode='full'
    )
    layerSh4_output = layerSh4.output[:, :, 1:-1, 1:-1]
    # layerSh4_output is now of shape (#batch, #nkerns[7]=48, 10, 10)
    layerSh4.W.name = 'Conv_layerSh4_W'
    layerSh4.b.name = 'Conv_layerSh4_b'
    layerSh4 = layerSh4

    #################################
    # Double Coarse branch layers D #
    #################################
    num_branch += 1

    # first conv layer
    layerD1 = ConvPoolLayer(
        rng,
        dropout = dropout,
        input=layerSh4_output,
        image_shape=(None, nkerns[7], None, None),
        filter_shape=(nkerns[8], nkerns[7], conv_large, conv_large),
        act='relu',
        pool=False,
        border_mode='full'
    )
    layerD1.W.name = 'Conv_layerD1_W'
    layerD1.b.name = 'Conv_layerD1_b'
    layerD1 = layerD1
    layerD1_output = layerD1.output[:, :, conv_border:-conv_border, conv_border:-conv_border]
    # layerD1_output is now of shape (#batch, #nkerns[7]=48, 10, 10)

    # second conv layer
    layerD2 = ConvPoolLayer(
        rng,
        dropout = dropout,
        input=layerD1_output,
        image_shape=(None, nkerns[8], None, None),
        filter_shape=(nkerns[0], nkerns[8], conv_small, conv_small),
        act=act_final_branch,
        pool=False,
        border_mode='full'
    )
    layerD2.W.name = 'Conv_layerD2_W'
    layerD2.b.name = 'Conv_layerD2_b'
    layerD2 = layerD2
    if conv_small != 1:
        layerD2_output = layerD2.output[:, :, conv_border:-conv_border, conv_border:-conv_border]
    else:
        layerD2_output = layerD2.output
    # layerD2_output is now of shape (#batch, #nkerns[0]=5, 10, 10)

    ##########################
    # upsampling layer for D #
    ##########################
    ratio = 8
    if bilinear:
        zero_tensor = T.zeros_like(layerF2_output)
        # copying the values of the coarse model equally spaces to a zero matrix
        layerD3_input = T.set_subtensor(zero_tensor[:, :, ::ratio, ::ratio], layerD2_output)
        layerD3_filters = bilinear_weights(dim=nkerns[0], ratio=ratio).astype(theano.config.floatX)
        index_start = layerD3_filters.shape[-1]/2

        layerD3 = conv.conv2d(
            input=layerD3_input,
            filters=layerD3_filters,
            image_shape=(None, nkerns[0], None, None),
            filter_shape=layerD3_filters.shape,
            border_mode='full'
        )
        layerD3 = layerD3
        layerD3_output = layerD3[:, :, index_start: -index_start, index_start: -index_start]
        # layerD3_output is now of shape (#batch, #nkerns[0]=5, 80, 80)
    else:
        layerD3_upsample = T.extra_ops.repeat(layerD2_output, repeats=ratio, axis=-1)
        layerD3_output = T.extra_ops.repeat(layerD3_upsample, repeats=ratio, axis=-2)

    ######################
    ### shared layers5 ###
    ######################
    # shared pool layer
    layerP4 = PoolLayer(
        input=layerSh4_output,
        dropout = dropout,
        poolsize=(2,2),
        ignore_border=False
    )
    layerP4_output = layerP4.output
    # layerP4_output is now of shape (#batch, #nkerns[7]=48, 5, 5)
    layerP4 = layerP4

    # shared conv layer
    layerSh5 = ConvPoolLayer(
        rng,
        dropout = dropout,
        input=layerP4_output,
        image_shape=(None, nkerns[7], None, None),
        filter_shape=(nkerns[9], nkerns[7], 3, 3),
        act='relu',
        pool=False,
        border_mode='full'
    )
    layerSh5_output = layerSh5.output[:, :, 1:-1, 1:-1]
    # layerSh5_output is now of shape (#batch, #nkerns[9]=96, 5, 5)
    layerSh5.W.name = 'Conv_layerSh5_W'
    layerSh5.b.name = 'Conv_layerSh5_b'
    layerSh5 = layerSh5

    ################################
    # Super Coarse branch layers S #
    ################################
    num_branch += 1

    # first conv layer
    layerS1 = ConvPoolLayer(
        rng,
        dropout = dropout,
        input=layerSh5_output,
        image_shape=(None, nkerns[9], None, None),
        filter_shape=(nkerns[10], nkerns[9], conv_large, conv_large),
        act='relu',
        pool=False,
        border_mode='full'
    )
    layerS1.W.name = 'Conv_layerS1_W'
    layerS1.b.name = 'Conv_layerS1_b'
    layerS1 = layerS1
    layerS1_output = layerS1.output[:, :, conv_border:-conv_border, conv_border:-conv_border]
    # layerS1_output is now of shape (#batch, #nkerns[10]=96, 5, 5)

    sys.stderr.write("coarsest layer conv size is %s.\n" %(coarse_conv_size,))
    # second conv layer
    layerS2 = ConvPoolLayer(
        rng,
        dropout = dropout,
        input=layerS1_output,
        image_shape=(None, nkerns[10], None, None),
        filter_shape=(nkerns[0], nkerns[10], conv_small, conv_small),
        act=act_final_branch,
        pool=False,
        border_mode='full'
    )
    layerS2.W.name = 'Conv_layerS2_W'
    layerS2.b.name = 'Conv_layerS2_b'
    layerS2 = layerS2
    border = coarse_conv_size/2
    if conv_small != 1:
        layerS2_output = layerS2.output[:, :, conv_border:-conv_border, conv_border:-conv_border]
    else:
        layerS2_output = layerS2.output
    # layerS2_output is now of shape (#batch, #nkerns[0]=5, 5, 5)

    ##########################
    # upsampling layer for S #
    ##########################
    ratio = 16
    if bilinear:
        zero_tensor = T.zeros_like(layerF2_output)
        # copying the values of the coarse model equally spaces to a zero matrix
        layerS3_input = T.set_subtensor(zero_tensor[:, :, ::ratio, ::ratio], layerS2_output)
        layerS3_filters = bilinear_weights(dim=nkerns[0], ratio=ratio).astype(theano.config.floatX)
        index_start = layerS3_filters.shape[-1]/2

        layerS3 = conv.conv2d(
            input=layerS3_input,
            filters=layerS3_filters,
            image_shape=(None, nkerns[0], None, None),
            filter_shape=layerS3_filters.shape,
            border_mode='full'
        )
        layerS3 = layerS3
        layerS3_output = layerS3[:, :, index_start: -index_start, index_start: -index_start]
        # layerS3_output is now of shape (#batch, #nkerns[0]=5, 80, 80)
    else:
        layerS3_upsample = T.extra_ops.repeat(layerS2_output, repeats=ratio, axis=-1)
        layerS3_output = T.extra_ops.repeat(layerS3_upsample, repeats=ratio, axis=-2)

    ######################
    ### shared layers6 ###
    ######################
    # shared pool layer
    if use_res_2:
        layerP5 = PoolLayer(
            input=layerSh5_output,
            dropout = dropout,
            poolsize=(3,3),
            stride_size=(2,2),
            ignore_border=True
        )
        layerP5_output = layerP5.output
        # layerP5_output is now of shape (#batch, #nkerns[10]=48, 2, 2)
        layerP5 = layerP5

        # shared conv layer
        layerSh6 = ConvPoolLayer(
            rng,
            dropout = dropout,
            input=layerP5_output,
            image_shape=(None, nkerns[10], None, None),
            filter_shape=(nkerns[10], nkerns[10], 3, 3),
            act='relu',
            pool=False,
            border_mode='full'
        )
        layerSh6_output = layerSh6.output[:, :, 1:-1, 1:-1]
        # layerSh6_output is now of shape (#batch, #nkerns[10]=48, 2, 2)
        layerSh6.W.name = 'Conv_layerSh6_W'
        layerSh6.b.name = 'Conv_layerSh6_b'
        layerSh6 = layerSh6

        layerT1_input = layerSh6_output
        layerT1_concat_dim = 0

        ##############################
        # 2*2 Corase branch layers T #
        ##############################
        num_branch += 1

        # first conv layer
        layerT1 = ConvPoolLayer(
            rng,
            dropout = dropout,
            input = layerT1_input,
            image_shape = (None, nkerns[10]+layerT1_concat_dim, None, None),
            filter_shape = (nkerns[10], nkerns[10]+layerT1_concat_dim, conv_large, conv_large),
            act='relu',
            pool=False,
            border_mode='full'
        )
        layerT1.W.name = 'Conv_layerT1_W'
        layerT1.b.name = 'Conv_layerT1_b'
        layerT1 = layerT1
        layerT1_output = layerT1.output[:, :, conv_border:-conv_border, conv_border:-conv_border]
        # layerT1_output is now of shape (#batch, #nkerns[10]=48, 2, 2)

        # second conv layer
        layerT2 = ConvPoolLayer(
            rng,
            dropout = dropout,
            input=layerT1_output,
            image_shape=(None, nkerns[10], None, None),
            filter_shape=(nkerns[0], nkerns[10], conv_small, conv_small),
            act=act_final_branch,
            pool=False,
            border_mode='full'
        )
        layerT2.W.name = 'Conv_layerT2_W'
        layerT2.b.name = 'Conv_layerT2_b'
        layerT2 = layerT2
        if conv_small != 1:
            layerT2_output = layerT2.output[:, :, conv_border:-conv_border, conv_border:-conv_border]
        else:
            layerT2_output = layerT2.output
        # layerT2_output is now of shape (#batch, #nkerns[10]=48, 2, 2)

        ##########################
        # upsampling layer for T #
        ##########################
        ratio = 40
        if bilinear:
            zero_tensor = T.zeros_like(layerT2_output)
            # copying the values of the coarse model equally spaces to a zero matrix
            layerT3_input = T.set_subtensor(zero_tensor[:, :, ::ratio, ::ratio], layerT2_output)
            layerT3_filters = bilinear_weights(dim=nkerns[0], ratio=ratio).astype(theano.config.floatX)
            index_start = layerT3_filters.shape[-1]/2

            layerT3 = conv.conv2d(
                input=layerT3_input,
                filters=layerT3_filters,
                image_shape=(None, nkerns[0], None, None),
                filter_shape=layerT3_filters.shape,
                border_mode='full'
            )
            layerT3 = layerT3
            layerT3_output = layerT3[:, :, index_start: -index_start, index_start: -index_start]
            # layerT3_output is now of shape (#batch, #nkerns[0]=5, 80, 80)
        else:
            layerT3_upsample = T.extra_ops.repeat(layerT2_output, repeats=ratio, axis=-1)
            layerT3_output = T.extra_ops.repeat(layerT3_upsample, repeats=ratio, axis=-2)

    ##############################
    # buildign the softmax layer #
    ##############################
    # concat_branch is of shape (#branch, #batch, #kpts, #dim, #dim)
    concat_branch = [layerFOut_output, layerM3_output, layerC3_output, layerD3_output, layerS3_output]
    if use_res_2:
        concat_branch.extend([layerT3_output])

    if weight_per_pixel:
        # in this case, every pixel location in 2D feature map
        # (for each kpts in each branch) is given a weight
        # W_branch is of shape (#branch, #kpts, #dim, #dim)
        W_branch = theano.shared(rng.uniform(size=(num_branch, num_keypoints, dim, dim)).astype(theano.config.floatX), name='W_branch')
        # for each branch a weight matrix of shape (#kpts, #dim, #dim) is used.
        conv_sum_concat = concat_branch * W_branch.dimshuffle(0, 'x', 1 , 2, 3)

    elif conv_per_kpt or linear_conv_per_kpt:
        # in this case, for each kpt a convolution is applied on
        # all feature outputs from different branches to get one map
        # for that kpt
        # linear_conv_per_kpt is similar to conv_per_kpt, with the difference
        # that the final conv in each branch (before upsampling) is also linear

        # concat_branch is of shape (#branch, #batch, #kpts, #dim, #dim)
        concat_branch = T.as_tensor(concat_branch)
        # kpt_arranged is of shape (#kpts, #batch, #branch, #dim, #dim)
        kpt_arranged = concat_branch.dimshuffle(2, 1, 0, 3, 4)
        layerKpts = []
        kpt_pre_softmax = []
        print "num_keypoints is %s" %(num_keypoints)
        for kpt in xrange(num_keypoints):
            # kpt_features is of shape (#batch, #branch, #dim, #dim)
            kpt_features = kpt_arranged[kpt]

            # conv layer on features of all branches for a keypoint
            layerKpt = ConvPoolLayer(
                rng,
                dropout = dropout,
                input=kpt_features,
                image_shape=(None, num_branch, None, None),
                filter_shape=(1, num_branch, conv_small, conv_small),
                act='linear',
                pool=False,
                border_mode='full'
            )
            layerKpt.W.name = 'Conv_layerKpt_W'
            layerKpt.b.name = 'Conv_layerKpt_b'
            layerKpts.append(layerKpt)
            if conv_small != 1:
                layerKpt_output = layerKpt.output[:, :, conv_border:-conv_border, conv_border:-conv_border]
            else:
                layerKpt_output = layerKpt.output
            # layerKpt_output is now of shape (#batch, 1, 80, 80)
            kpt_pre_softmax.append(layerKpt_output)

    else:
        # in this case every kpt filter in a branch gets a weight
        # W_branch is of shape (#branch, #kpts)
        W_branch = theano.shared(rng.uniform(size=(num_branch, num_keypoints)).astype(theano.config.floatX), name='W_branch')
        # conv_sum_concat is of shape (#branch, #batch, #kpts, #rows, #cols)
        conv_sum_concat = concat_branch * W_branch.dimshuffle(0, 'x', 1 ,'x', 'x')

    if conv_per_kpt or linear_conv_per_kpt:
        # conv_sum is of shape (#batch, #kpts, #rows, #cols)
        conv_sum = T.concatenate(kpt_pre_softmax, axis=1)
    else:
        # summing over different branches
        # conv_sum is of shape (#batch, #kpts, #rows, #cols)
        conv_sum = conv_sum_concat.sum(axis=0)

    # conv_sum_3D is of shape (#batch_size * #kpts, #rows, #cols)
    conv_sum_3D = conv_sum.reshape((-1, dim, dim))
    # unormalized_probs is of shape (#batch_size * #kpts, #row * #cols)
    # where for each batch example all keypoints are serialized before having the next
    # example in the first dimension (exp1_kpt1, exp1_kpt2,.., exp1_kptn, exp2_kpt1, ...)
    unormalized_probs = conv_sum_3D.flatten(2)
    softmax_layer = Softmax(unormalized_probs)

    #######################################
    # setting the parameters of the model #
    #######################################
    # create a list of all model parameters to be fit by gradient descent
    # defininig params of MTFL model
    params_common = layerSh1.params + layerSh2.params + layerSh3.params + layerSh4.params + layerSh5.params
    params_branch = layerF1.params + layerF2.params + layerF3.params + layerFOut.params + layerM1.params + layerM2.params +\
                    layerC1.params + layerC2.params + layerD1.params + layerS1.params + layerS2.params
    params_branch_no_output =  layerM1.params + layerM2.params +layerC1.params + layerC2.params +\
                               layerD1.params + layerS1.params + layerS2.params +\
                               layerF1.params + layerF2.params + layerF3.params

    if use_res_2:
        params_common += layerSh6.params
        params_branch += layerT1.params + layerT2.params
        params_branch_no_output += layerT1.params + layerT2.params

    if extra_fine:
        params_branch += layerF4.params + layerF5.params
        params_branch_no_output += layerF4.params + layerF5.params

    params = params_common + params_branch
    params_no_output = params_common + params_branch_no_output

    #########################
    # measuring the L2 cost #
    #########################
    # L2 norm for common convolutional layers
    L2_sqr_common = ((layerSh1.W ** 2).sum() + (layerSh2.W ** 2).sum() + (layerSh3.W ** 2).sum() + (layerSh4.W ** 2).sum() +\
                     (layerSh5.W ** 2).sum())
    L2_sqr_branch = (layerF1.W ** 2).sum() + (layerF2.W ** 2).sum() + (layerF3.W ** 2).sum() + (layerFOut.W ** 2).sum() +\
                    (layerM1.W ** 2).sum() + (layerM2.W ** 2).sum() +\
                    (layerC1.W ** 2).sum() + (layerC2.W ** 2).sum() + (layerD1.W ** 2).sum() +\
                    (layerS1.W ** 2).sum() + (layerS2.W ** 2).sum()

    if use_res_2:
        L2_sqr_common += (layerSh6.W ** 2).sum()
        L2_sqr_branch += (layerT1.W ** 2).sum() + (layerT2.W ** 2).sum()

    if extra_fine:
        L2_sqr_branch += (layerF4.W ** 2).sum() + (layerF5.W ** 2).sum()

    return conv_sum, softmax_layer, params, params_no_output, L2_sqr_common, L2_sqr_branch


class TCDCN_ConvNet(object):
    def __init__(self, learning_rate, use_ada_delta, decay, train_cost, num_img_channels, nkerns,
                 param_seed, mask_MTFL_layer, mask_300W_layer, use_lcn, target_dim=80,
                 bilinear=False, bch_norm=False, use_res_2=False, extra_fine=False,
                 coarse_conv_size=3, conv_per_kpt=False, linear_conv_per_kpt=False,
                 weight_per_pixel=False, **kwargs):
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        dim = target_dim
        num_keypoints = 68
        nkerns[0] = num_keypoints
        sys.stderr.write("number of channels is %i\n" %num_img_channels)
        if use_res_2:
            sys.stderr.write("using conv up to resolution 2\n")

        rng = np.random.RandomState(param_seed)
        sys.stderr.write("rng seed for parameter initialization is %i\n" %param_seed)

        #theano.config.compute_test_value = 'raise'
        sys.stderr.write('... building the model\n')
        x = T.tensor4('x', dtype=theano.config.floatX)  # the data is presented in 4D shape (#batch, #row, #cols, , #channels)
        # the given input is the shape (#batch, #row, #cols, #channels)
        # in order to pass it to the first convlolutional layer it should be reshaped to
        # (#batch, #channels, #row, #cols)
        x.tag.test_value = np.random.rand(128, dim, dim, num_img_channels).astype(theano.config.floatX)
        x_input = x.dimshuffle(0, 3, 1, 2)
        if use_lcn:
            sys.stderr.write("using local contrast normalization\n")
            x_input = lcn(x_input, num_img_channels, dim)

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
        L2_coef_common = T.scalar('L2_coef_common', dtype=theano.config.floatX)
        L2_coef_common.tag.test_value = np.float32(1.0)
        # L2 cost coefficient for the fully connected layer
        L2_coef_branch = T.scalar('L2_coef_branch', dtype=theano.config.floatX)
        L2_coef_branch.tag.test_value = np.float32(1.0)

        # mask_border is a vector of shape (#batch * #kpts) indicating
        # for each sample which keypoint is on the border.
        # the order of values is: (exp1_kpt1, exp1_kpt2,.., exp1_kptn, exp2_kpt1, ...)
        # mask_border is one when the kpt is not in the pad region and zero otherwise.
        mask_border = T.vector('mask_border', dtype=theano.config.floatX)
        mask_border.tag.test_value = (np.ones((128 * num_keypoints))).astype(theano.config.floatX)

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
        layer1_input = x_input

        dropout = T.scalar('dropout', dtype=theano.config.floatX)
        dropout.tag.test_value = np.float32(0.)

        conv_sum, softmax_layer, params, params_no_output, L2_sqr_common, L2_sqr_branch = fforward_model(
            layer1_input=layer1_input, dropout=dropout, bch_norm=bch_norm,
            nkerns=nkerns, num_img_channels=num_img_channels, dim=dim, rng=rng,
            use_res_2=use_res_2, bilinear=bilinear, coarse_conv_size=coarse_conv_size,
            conv_per_kpt=conv_per_kpt, linear_conv_per_kpt=linear_conv_per_kpt,
            weight_per_pixel=weight_per_pixel, extra_fine=extra_fine)

        self.params = params
        self.params_no_output = params_no_output
        ########################################
        # getting the cost of the output layer #
        ########################################
        epsilon = 1e-8
        # cost_batch is a vector of dim (#batch_size * #kpts)
        cost_batch = softmax_layer.negative_log_likelihood(y_kpt_norm_serial)
        cost_kpt = T.sum(cost_batch * mask_border) / (T.sum(mask_border) + epsilon)
        # cost is the sum of the cost of the keypoints
        cost_kpt *= num_keypoints

        ########################################################
        # getting the sum log probs for the detected locations #
        ########################################################
        softmax_probs = softmax_layer.p_y_given_x
        max_prob = T.max(softmax_probs, axis=1)
        log_max_prob = -T.log(max_prob)
        # log_max_prob_2D is of shape (#batch, #kpts)
        log_max_prob_2D = log_max_prob.reshape((-1, num_keypoints))
        sum_log_probs = T.sum(log_max_prob_2D, axis=1)

        ###########################################
        # getting the softmax output of the model #
        ###########################################
        softmax_output = softmax_probs.reshape((-1, num_keypoints, dim, dim))

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
        error_kpt_masked_train = kpt_euc_dist_train * mask_border

        # error_kpt_2D is of shape (#batch , #kpt)
        error_kpt_2D_train = error_kpt_masked_train.reshape((-1, num_keypoints))

        # the values of x_pred, y_pred, x_true, y_true
        # are in the range of [0,dim). So to make the
        # calculation compatible y_kpt_ocular should also
        # get unnormalized
        y_kpt_ocular_unorm = y_kpt_ocular * dim
        error_kpt_each_norm_MTFL_train = error_kpt_2D_train / y_kpt_ocular_unorm

        # getting the average error for each keypoint
        # error_kpt is a vector of shape (#keypoints)
        error_kpt_MTFL_train = T.sum(error_kpt_each_norm_MTFL_train, axis=0)

        # error_kpt_avg gets the average over all keypoints,
        # while sums over all
        error_kpt_avg_MTFL_train = T.mean(error_kpt_MTFL_train)

        error_kpt_train = error_kpt_MTFL_train
        error_kpt_avg_train = error_kpt_avg_MTFL_train

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

        x_diff_sqr = (x_pred - x_true)**2
        y_diff_sqr = (y_pred - y_true)**2
        # kpt_euc_dist is of shape (#batch * #kpt)
        kpt_euc_dist = T.sqrt(x_diff_sqr + y_diff_sqr)
        error_kpt_masked = kpt_euc_dist

        # error_kpt_2D is of shape (#batch , #kpt)
        error_kpt_2D = error_kpt_masked.reshape((-1, num_keypoints))

        # the values of x_pred, y_pred, x_true, y_true
        # are in the range of [0,dim). So to make the
        # calculation compatible y_kpt_ocular should also
        # get unnormalized
        y_kpt_ocular_unorm = y_kpt_ocular * dim
        error_kpt_each_norm_MTFL = error_kpt_2D / y_kpt_ocular_unorm

        # getting the average error for each keypoint
        # error_kpt is a vector of shape (#keypoints)
        error_kpt_MTFL = T.sum(error_kpt_each_norm_MTFL, axis=0)

        # error_kpt_avg gets the average over all keypoints
        error_kpt_avg_MTFL = T.mean(error_kpt_MTFL)

        error_kpt = error_kpt_MTFL
        error_kpt_avg = error_kpt_avg_MTFL

        #############################
        # getting errors seperately #
        #############################
        # error_kpt_each contains the error seperately for each batch_sample
        error_kpt_each = T.mean(error_kpt_each_norm_MTFL, axis=1)

        # the same variable for the train set
        error_kpt_each_train = T.mean(error_kpt_each_norm_MTFL_train, axis=1)

        L2_cost = L2_coef_common * L2_sqr_common + L2_coef_branch * L2_sqr_branch

        cost = L2_cost + cost_kpt

        #######################################
        # defining the optimization algorithm #
        #######################################

        # specifying the parameter update algorithm
        self.tr = None
        if use_ada_delta:
            # setting the updates using the ada_delta
            self.tr = Train_alg()
            updates = self.tr.build_updates(cost=cost, params=self.params, consider_constant=None, decay=decay)
        else: # using normal update procedure with a fixed learning rate
            # create a list of gradients for all model parameters
            sys.stderr.write("using normal stochastic gradient descent.\n")
            grads = T.grad(cost, self.params)

            # train_model is a function that updates the model parameters by
            # SGD Since this model has many parameters, it would be tedious to
            # manually create an update rule for each model parameter. We thus
            # create the updates list by automatically looping over all
            # (params[i], grads[i]) pairs.
            updates = [
                (param_i, param_i - learning_rate * grad_i)
                for param_i, grad_i in zip(self.params, grads)
            ]

        ###############################
        # defining the test functions #
        ###############################
        self.train_model = theano.function(
            [L2_coef_common, L2_coef_branch, x, y_kpt_ocular_dist, y_kpt_norm, mask_border, dropout],
            [cost, cost_kpt, L2_cost, error_kpt_train, error_kpt_avg_train],
            updates=updates, allow_input_downcast=True)

        self.valid_model = theano.function(
            [L2_coef_common, L2_coef_branch, x, y_kpt_ocular_dist, y_kpt_norm, mask_border, dropout],
            [cost, cost_kpt, L2_cost, error_kpt_train, error_kpt_avg_train], allow_input_downcast=True,
            on_unused_input='ignore')

        # testing only on MTFL dataset with no task usage
        self.test_300W = theano.function(
            [L2_coef_common, L2_coef_branch, x, y_kpt_ocular_dist, y_kpt_norm, bound_mask, mask_border, border_pixel, dropout],
            [cost_kpt, L2_cost, error_kpt, error_kpt_avg], allow_input_downcast=True)

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

        self.get_keypoints_MTFL = theano.function([x, bound_mask, border_pixel, dropout], kpt_pred_300W,
                                                  allow_input_downcast=True)
        self.get_keypoints_MTFL_train = theano.function([x, dropout], kpt_pred_300W_train,
                                                        allow_input_downcast=True)

        self.get_softmax_output = theano.function([x, dropout], softmax_output, allow_input_downcast=True)

        self.get_pre_softmax = theano.function([x, dropout], conv_sum, allow_input_downcast=True)

        # this method returns 2D vector of shape (#batch, #kpts), with each value in the range [0, dim**2)
        self.model_prediction = theano.function([x, dropout], predict_2D, allow_input_downcast=True)

        self.get_errors = theano.function([x, y_kpt_ocular_dist, y_kpt_norm, bound_mask, border_pixel, dropout],
                                          error_kpt_each, allow_input_downcast=True)

        self.get_errors_train = theano.function([x, y_kpt_ocular_dist, y_kpt_norm, mask_border, dropout],
                                                error_kpt_each_train, allow_input_downcast=True)

        self.get_sum_log_probs = theano.function([x, dropout], [sum_log_probs, log_max_prob_2D, max_prob],
                                                 allow_input_downcast=True)


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

    def load_params(self, pickle_path, load_adedelta=False, load_no_output_params=False):
        # complete pickle path should be given
        "This method loads the parameters of the TCDCN_ConvNet from a pickle file"
        #pickle_param_file = dest_dir + '/' + pickle_name
        pickle_param_file = pickle_path

        # if load_no_output_params==True, then only load the model params
        # excluding the output layer
        if load_no_output_params:
            model_params = self.params_no_output
        else:
            model_params = self.params

        with open (pickle_param_file, 'rb') as fp:
            for param in model_params:
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
    def __init__(self, data_queue, seed_queue, nkerns, num_epochs, learning_rate, batch_size, sliding_window_lenght, task_stop_threshold,
                 L2_coef_common, L2_coef_branch, use_ada_delta, decay, param_path, train_cost, file_suffix, num_img_channels, sets,
                 param_seed, num_procs, Lambda_coefs, mask_MTFL, mask_300W, use_lcn, producers, sw_lenght, target_dim, bilinear, bch_norm,
                 dropout, num_queue_elem, use_res_2, extra_fine, load_no_output_params, coarse_conv_size, conv_per_kpt, linear_conv_per_kpt,
                 weight_per_pixel):

        if '300W' in data_queue.keys():
            self.data_queue_300W = data_queue['300W']
        if 'MTFL' in data_queue.keys():
            self.data_queue_MTFL = data_queue['MTFL']
        self.seed_queue = seed_queue
        self.num_procs = num_procs

        tcdcn = TCDCN_ConvNet(learning_rate=learning_rate, use_ada_delta=use_ada_delta, decay=decay, train_cost=train_cost,
                              num_img_channels=num_img_channels, nkerns=nkerns, param_seed=param_seed, mask_MTFL_layer=mask_MTFL,
                              mask_300W_layer=mask_300W, use_lcn=use_lcn, target_dim=target_dim, bilinear=bilinear,
                              bch_norm=bch_norm, use_res_2=use_res_2, extra_fine=extra_fine, coarse_conv_size=coarse_conv_size,
                              conv_per_kpt=conv_per_kpt, linear_conv_per_kpt=linear_conv_per_kpt, weight_per_pixel=weight_per_pixel)

        ####################################
        # running a previously saved_model #
        ####################################
        if param_path != "":
            # if param_path is given the params are loaded from file
            if load_no_output_params:
                print "loading params (exclusing output layer) from %s" %param_path
                # loading params of a model excluding the output layer and
                # excluding the adadelta params
                tcdcn.load_params(param_path, load_no_output_params=True)

            else:
                # loading params of the entire model
                print "loading params from %s" %param_path
                tcdcn.load_params(param_path, load_adedelta=True)

        self.num_queue_elem = num_queue_elem
        self.EOF_used = False
        self.tcdcn = tcdcn
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.sliding_window_lenght = sliding_window_lenght
        self.task_stop_threshold = task_stop_threshold
        self.L2_coef_common = L2_coef_common
        self.L2_coef_branch = L2_coef_branch
        self.file_suffix = file_suffix
        self.sets = sets
        self.Lambda_coefs = Lambda_coefs
        self.mask_MTFL = mask_MTFL
        self.mask_300W = mask_300W
        self.producers = producers
        self.sw_lenght = sw_lenght
        self.dim = target_dim
        self.bch_norm = bch_norm
        self.dropout = dropout
        self.bch_norm_bs = 256

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

            # getting border_pixels
            pad_ratio = set_y['pad_ratio']
            # change pad_ratio to border pixel locations
            border_pixel = padRatio_to_pixels(pad_ratio, self.train_set_x_300W.shape[1])
            set_y['border_pixel'] = border_pixel

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
        y_bound_mask = self.train_set_y_300W['bound_mask'][index_300W * batch_size: (index_300W + 1) * batch_size]
        y_pad_ratio = self.train_set_y_300W['pad_ratio'][index_300W * batch_size: (index_300W + 1) * batch_size]
        y_mask_border = self.train_set_y_300W['mask_border'][index_300W * batch_size: (index_300W + 1) * batch_size]
        y_border_pixel = self.train_set_y_300W['border_pixel'][index_300W * batch_size: (index_300W + 1) * batch_size]

        # updating index values
        self.index_300W = (index_300W + 1) % self.train_nbatches_300W

        # updating the total number of seen examples
        self.samples_seen += x.shape[0]

        #y_mask_border is of shape (#batch * #kpts)
        y_mask_border = np.ndarray.flatten(y_mask_border)

        return [x, y_kpt_norm, y_ocular_dist, y_bound_mask, y_mask_border, y_border_pixel]


    def get_mini_batch_valid_300W(self, valid_set_x, valid_set_y, index, batch_size):
        # this method gets the next mini-batch only for one datast.
        x = valid_set_x[index * batch_size: (index + 1) * batch_size]
        y_ocular_dist = valid_set_y['ocular_dist'][index * batch_size: (index + 1) * batch_size]
        y_kpt_norm = valid_set_y['kpt_norm'][index * batch_size: (index + 1) * batch_size]
        y_bound_mask = valid_set_y['bound_mask'][index * batch_size: (index + 1) * batch_size]
        y_pad_ratio = valid_set_y['pad_ratio'][index * batch_size: (index + 1) * batch_size]
        y_mask_border = valid_set_y['mask_border'][index * batch_size: (index + 1) * batch_size]
        y_border_pixel = valid_set_y['border_pixel'][index * batch_size: (index + 1) * batch_size]
        y_mask_border = np.ndarray.flatten(y_mask_border)
        return [x, y_kpt_norm, y_ocular_dist, y_bound_mask, y_mask_border, y_border_pixel]

    def append_errors(self, error_dict, epoch_sets, epoch, num_samples, is_train=True, batch_sets=None):
        epoch_cost, epoch_cost_kpt, epoch_error_kpt, epoch_error_kpt_avg, epoch_l2_cost = epoch_sets
        sw_lenght = self.sw_lenght
        this_epoch_cost = np.mean(epoch_cost)
        error_dict['cost'].append(this_epoch_cost)
        this_epoch_cost_kpt = np.mean(epoch_cost_kpt)
        error_dict['cost_kpt'].append(this_epoch_cost_kpt)
        epoch_error_kpt = np.sum(np.array(epoch_error_kpt), axis=0)
        error_dict['error_kpt'].append(epoch_error_kpt/num_samples)
        this_epoch_error_kpt_avg = np.sum(epoch_error_kpt_avg)/num_samples
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
        test_num_samples = error_dict['num_samples']
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
                cost_kpt, L2_cost, error_kpt, error_kpt_avg = self.tcdcn.test_MTFL(self.L2_coef_common, self.L2_coef_branch,\
                             x, y_kpt_ocular_dist, y_kpt_norm, mask_MTFL, dropout=0)
            else:
                # adding 300W data
                x = test_set_x[index * batch_size: (index + 1) * batch_size]
                y_kpt_norm = test_set_y['kpt_norm'][index * batch_size: (index + 1) * batch_size]
                y_kpt_ocular_dist = test_set_y['ocular_dist'][index * batch_size: (index + 1) * batch_size]
                y_bound_mask = test_set_y['bound_mask'][index * batch_size: (index + 1) * batch_size]
                #y_pad_ratio = test_set_y['pad_ratio'][index * batch_size: (index + 1) * batch_size]
                y_mask_border = test_set_y['mask_border'][index * batch_size: (index + 1) * batch_size]
                y_border_pixel = test_set_y['border_pixel'][index * batch_size: (index + 1) * batch_size]
                #y_mask_border is of shape (#batch * #kpts)
                y_mask_border = np.ndarray.flatten(y_mask_border)
                cost_kpt, L2_cost, error_kpt, error_kpt_avg = self.tcdcn.test_300W(self.L2_coef_common, self.L2_coef_branch,\
                             x, y_kpt_ocular_dist, y_kpt_norm, y_bound_mask, y_mask_border, y_border_pixel, dropout=0)

            # accumulating the values of the mini-batcjes
            epoch_cost_kpt.append(cost_kpt)
            epoch_error_kpt.append(error_kpt)
            epoch_error_kpt_avg.append(error_kpt_avg)

        # getting the average of the whole epoch
        avg_epoch_cost_kpt = np.mean(epoch_cost_kpt)
        # getting the average of each keypoint over all of the samples
        epoch_error_kpt = np.sum(np.array(epoch_error_kpt), axis=0)
        avg_epoch_error_kpt = epoch_error_kpt/test_num_samples
        avg_epoch_error_kpt_avg = np.sum(epoch_error_kpt_avg)/test_num_samples
        # appending epoch results
        error_dict['cost_kpt'].append(avg_epoch_cost_kpt)
        error_dict['error_kpt'].append(avg_epoch_error_kpt)
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
        sliding_window_lenght = self.sliding_window_lenght
        task_stop_threshold = self.task_stop_threshold
        L2_coef_common = self.L2_coef_common
        L2_coef_branch = self.L2_coef_branch
        file_suffix = self.file_suffix
        Lambda_coefs = self.Lambda_coefs
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

        # creating dummy values for the case when one of the datasets is not used in training
        self.y_kpt_MTFL_dummy = np.zeros((self.batch_size, self.num_kpts_MTFL * 2))
        self.y_kpt_300W_dummy = np.zeros((self.batch_size, self.num_kpts_300W * 2))
        self.y_task_dummy = np.zeros((self.batch_size))

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
        for epoch in xrange(num_epochs):
            # checking whether child processes are stil alive
            if self.producers:
                for pr in self.producers:
                    if pr.exitcode > 0:
                        sys.stderr.write("An error encountered in one of the child processes. exiting ...%i\n")
                        exit()
            epoch_start_time = time.time()
            epoch_cost = []
            epoch_error_test = []
            epoch_cost_kpt = []
            epoch_error_kpt = []
            epoch_error_kpt_avg = []
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
                x, y_kpt_norm, y_kpt_ocular_dist, y_bound_mask, y_mask_border, border_pixel = self.get_mini_batch_train_300W()
                # getting values in the range of [0, dim**2]
                if epoch == 0:
                    # in the first epoch, we just evaluate the performance of random initialization without any parameter update
                    # note that since the model is not trained in the first epoch, the valid and test sets cost and errors for the first epoch
                    # would be the model's performace before training
                    cost, cost_kpt, L2_cost, error_kpt, error_kpt_avg = tcdcn.valid_model(L2_coef_common, L2_coef_branch, x,
                                                                                y_kpt_ocular_dist, y_kpt_norm,
                                                                                y_mask_border, dropout=0)
                else:
                    cost, cost_kpt, L2_cost, error_kpt, error_kpt_avg = tcdcn.train_model(L2_coef_common, L2_coef_branch, x,
                                                                                y_kpt_ocular_dist, y_kpt_norm,
                                                                                y_mask_border, dropout=self.dropout)

                epoch_cost.append(cost)
                epoch_cost_kpt.append(cost_kpt)
                epoch_l2_cost.append(L2_cost)
                epoch_error_kpt.append(error_kpt)
                epoch_error_kpt_avg.append(error_kpt_avg)
                """
                epoch_cost_gl.append(cost_gl)
                epoch_cost_gen.append(cost_gen)
                epoch_cost_sm.append(cost_sm)
                epoch_cost_pose.append(cost_pose)
                epoch_lambda_gl.append(lambda_gl)
                epoch_lambda_gen.append(lambda_gen)
                epoch_lambda_sm.append(lambda_sm)
                epoch_lambda_pose.append(lambda_pose)
                """

            # appending epoch results
            epoch_sets = [epoch_cost, epoch_cost_kpt, epoch_error_kpt, epoch_error_kpt_avg, epoch_l2_cost]
            #######################################################
            # saving mini-batch logs only for the first 50 epochs #
            #######################################################
            if epoch < 50:
                batch_sets = [epoch_cost, epoch_cost_kpt, epoch_l2_cost]
            else:
                batch_sets = None
            num_samples = self.samples_seen
            train_epoch_cost, train_epoch_cost_kpt = self.append_errors(error_dict=Train_error, epoch_sets=epoch_sets, epoch=epoch,\
                          num_samples=num_samples, is_train=True, batch_sets=batch_sets)

            ################################
            # getting the valid set error #
            ###############################
            for subset in Valid.keys():
                setx, sety = Valid[subset]
                valid_error = Valid_error[subset]
                num_batches = valid_error['num_batches']
                num_samples = valid_error['num_samples']
                batch_size = self.batch_size
                epoch_cost = []
                epoch_error_test = []
                epoch_cost_kpt = []
                epoch_error_kpt = []
                epoch_error_kpt_avg = []
                epoch_cost_gl = []
                epoch_cost_gen = []
                epoch_cost_sm = []
                epoch_cost_pose = []
                epoch_l2_cost = []
                for index in np.arange(num_batches):
                    # getting the next mini-batch for the validation set
                    x, y_kpt_norm, y_kpt_ocular_dist, y_bound_mask, y_mask_border, border_pixel =\
                        self.get_mini_batch_valid_300W(setx, sety, index, batch_size)
                    # evaluating performance on the validation batch
                    cost, cost_kpt, L2_cost, error_kpt, error_kpt_avg = tcdcn.valid_model(L2_coef_common, L2_coef_branch, x,
                                                                                    y_kpt_ocular_dist, y_kpt_norm,
                                                                                    y_mask_border, dropout=0)
                    epoch_cost.append(cost)
                    epoch_cost_kpt.append(cost_kpt)
                    epoch_error_kpt.append(error_kpt)
                    epoch_error_kpt_avg.append(error_kpt_avg)
                    epoch_l2_cost.append(L2_cost)
                    """
                    epoch_cost_gl.append(cost_gl)
                    epoch_cost_gen.append(cost_gen)
                    epoch_cost_sm.append(cost_sm)
                    epoch_cost_pose.append(cost_pose)
                    """

                # appending epoch results
                epoch_sets = [epoch_cost, epoch_cost_kpt, epoch_error_kpt, epoch_error_kpt_avg, epoch_l2_cost]
                batch_sets = None
                num_samples = Valid_error[subset]['num_samples']
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
                # writing the epoch number to a txt file
                text_file_name = params_pickle_base + file_suffix + '.txt'
                text = "saved model params in epoch %i\n" %epoch
                append_text(text_file_name, text)
                period_start_time = current_time

            if (epoch+1) % 50 == 0 :
                sys.stderr.write("done with epoch %i\n" %(epoch + 1))
                epoch_100_end_time = time.time() # start time for training
                epoch_training_time = (epoch_100_end_time - epoch_100_start_time)
                sys.stderr.write('50 epochs took %f minutes\n' % (epoch_training_time / 60.))
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
