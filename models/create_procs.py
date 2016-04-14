import numpy as np
from collections import OrderedDict
import cPickle as pickle
import time
import sys
sys.setrecursionlimit(2000)
import argparse
import RCN
from RCN.preprocessing.tools import EOF
from RCN.preprocessing.tools import shuffleData, splitData, mergeData
from RCN.preprocessing.preprocess import preprocess_iter, preprocess_once
from RCN.utils.queue import OrderedQueue
from multiprocessing import Process, Queue
import os
import string

source_dir = os.path.dirname(RCN.__file__)
dest_dir = source_dir + '/models/exp_shared_conv'

def get_data(mask_MTFL, mask_300W, all_train, **kwargs):
    sys.stderr.write("\nloading data ...\n")
    Train = OrderedDict()
    Valid = OrderedDict()
    Test = OrderedDict()

    if mask_MTFL:
        # getting the MTFL train, valid and AFLW test set
        print "using 160by160 MTFL"
        MTFL_train = source_dir + '/datasets/MTFL_raw/MTFL_train_160by160.pickle'
        MTFL_test = source_dir + '/datasets/MTFL_raw/MTFL_test_160by160.pickle'
        # getting the AFW test set
        print "using 160by160 AFW"
        AFW_test = source_dir + '/datasets/MTFL_raw/AFW_test_160by160.pickle'
        sys.stderr.write("\nloading MTFL_train ...\n")

        # loading data
        with open(MTFL_train, 'rb') as fp:
            train_set = pickle.load(fp)
        sys.stderr.write("\nloading MTFL_test ...\n")
        with open(MTFL_test, 'rb') as fp:
            test_set = pickle.load(fp)
        sys.stderr.write("\nloading AFW_test ...\n")
        with open(AFW_test, 'rb') as fp:
            afw_set = pickle.load(fp)

        # get sets X and Y
        raw_set_x = train_set['X']
        raw_set_y = train_set['Y']
        test_set_x = test_set['X']
        test_set_y = test_set['Y']
        afw_set_x = afw_set['X']
        afw_set_y = afw_set['Y']

        # shuffle the data and split between train and validation sets
        shuffled_x, shuffled_y = shuffleData(raw_set_x, raw_set_y)
        train_set_x, train_set_y, valid_set_x, valid_set_y = splitData(shuffled_x, shuffled_y, split_size=1000)

        if all_train:
            print "using all train set for MTFL"
            train_set_x = shuffled_x
            train_set_y = shuffled_y

        # adding data to the sets
        Train['MTFL'] = (train_set_x, train_set_y)
        Valid['MTFL'] = (valid_set_x, valid_set_y)
        MTFL_test = OrderedDict()
        MTFL_test['aflw'] = (test_set_x, test_set_y)
        MTFL_test['afw'] = (afw_set_x, afw_set_y)
        Test['MTFL'] = MTFL_test

    if mask_300W:
        # getting the 300W train set
        print "using 160by160 300W"
        train_300W = source_dir + '/datasets/300W/300W_train_160by160.pickle'
        test_300W = source_dir + '/datasets/300W/300W_test_160by160.pickle'

        # loading data
        sys.stderr.write("\nloading 300W_train ...\n")
        with open(train_300W, 'rb') as fp:
            train_set_300W = pickle.load(fp)
        sys.stderr.write("\nloading 300W_test ...\n")
        with open(test_300W, 'rb') as fp:
            test_set_300W = pickle.load(fp)

        # getting sets X and Y
        train_set_300W_x = train_set_300W['X']
        train_set_300W_y = train_set_300W['Y']
        Helen_set = test_set_300W['Helen']
        Helen_set_x = Helen_set['X']
        Helen_set_y = Helen_set['Y']
        ibug_set = test_set_300W['ibug']
        ibug_set_x = ibug_set['X']
        ibug_set_y = ibug_set['Y']
        lfpw_set = test_set_300W['lfpw']
        lfpw_set_x = lfpw_set['X']
        lfpw_set_y = lfpw_set['Y']

        sys.stderr.write("data loaded.\n")
        # shuffle the data and split between train and validation sets for 300W
        sz_300W = train_set_300W_x.shape[0]/10
        shuffled_300W_x, shuffled_300W_y = shuffleData(train_set_300W_x, train_set_300W_y)
        train_300W_x, train_300W_y, valid_300W_x, valid_300W_y = splitData(shuffled_300W_x, shuffled_300W_y, split_size=sz_300W)

        if all_train:
            print "using all train set for 300W"
            train_300W_x = shuffled_300W_x
            train_300W_y = shuffled_300W_y

        # adding data to the sets
        Train['300W'] = (train_300W_x, train_300W_y)
        Valid['300W'] = (valid_300W_x, valid_300W_y)
        W300_test = OrderedDict()
        W300_test['ibug'] = (ibug_set_x, ibug_set_y)
        W300_test['lfpw'] = (lfpw_set_x, lfpw_set_y)
        W300_test['Helen'] = (Helen_set_x, Helen_set_y)
        Test['300W'] = W300_test

    sets = [Train, Valid, Test]
    return sets

def preprocess_data(sets, mask_MTFL, mask_300W, scale_mul,
                    translate_mul, gray_scale, use_lcn, dist_ratio, target_dim=80, block_img=False,
                    fixed_block=False, rotation=10):
    td = (target_dim, target_dim)
    rng_seed  = np.random.RandomState(0)
    Train, Valid, Test = sets

    MTFL_ratio = 3.8/4.0

    rotation_set = None

    if mask_MTFL:
        # getting the data from the OrderedDict
        train_set_x, train_set_y = Train['MTFL']
        valid_set_x, valid_set_y = Valid['MTFL']
        MTFL_test = Test['MTFL']
        test_set_x, test_set_y = MTFL_test['aflw']
        afw_set_x, afw_set_y = MTFL_test['afw']
        ######################
        # preprocessing data #
        ######################
        #note that set_x before and after pre-processing is a 4D tensor of size (#sampels, #rows, #cols, #channels)
        # where #channels=1 if the image is gray-scaled
        # setting the images to gray-scale and detecting the bounding box
        print "\ndoing preprocess_once on train_set"
        train_set_x, train_set_y = preprocess_once(train_set_x, train_set_y, gray_scale=gray_scale, dist_ratio=MTFL_ratio)
        print "\ndoing preprocess_once on valid_set"
        valid_set_x, valid_set_y = preprocess_once(valid_set_x, valid_set_y, gray_scale=gray_scale, dist_ratio=MTFL_ratio)
        print "\ndoing preprocess_once on aflw test_set"
        test_set_x, test_set_y = preprocess_once(test_set_x, test_set_y, gray_scale=gray_scale, dist_ratio=MTFL_ratio)

        #downsampling the valid and test sets
        print "\ndoing preprocess_iter on valid_set"
        valid_set_x, valid_set_y = preprocess_iter(valid_set_x, valid_set_y, rng_seed, target_dim=td,
                                                   jitter=True, scale_mul=scale_mul, translate_mul=translate_mul,
                                                   sanity=False, use_lcn=use_lcn, dset='MTFL', block_img=block_img,
                                                   fixed_block=fixed_block, rotation=rotation)
        print "\ndoing preprocess_iter on aflw test_set"
        test_set_x, test_set_y = preprocess_iter(test_set_x, test_set_y, rng_seed, target_dim=td,
                                                 jitter=False,  scale_mul=0., translate_mul=0., sanity=False,
                                                 use_lcn=use_lcn, dset='MTFL', block_img=False,
                                                 rotation_set=rotation_set)
        # processing the afw test set
        print "\ndoing preprocess_once on afw test_set"
        afw_set_x, afw_set_y = preprocess_once(afw_set_x, afw_set_y, gray_scale=gray_scale, dist_ratio=MTFL_ratio)
        print "\ndoing preprocess_iter on afw test_set"
        afw_set_x, afw_set_y = preprocess_iter(afw_set_x, afw_set_y, rng_seed, target_dim=td,
                                               jitter=False, scale_mul=0., translate_mul=0., sanity=False,
                                               use_lcn=use_lcn, dset='MTFL', block_img=False,
                                               rotation_set=rotation_set)

        # adding data to the sets
        Train['MTFL'] = (train_set_x, train_set_y)
        Valid['MTFL'] = (valid_set_x, valid_set_y)
        MTFL_test = OrderedDict()
        MTFL_test['aflw'] = (test_set_x, test_set_y)
        MTFL_test['afw'] = (afw_set_x, afw_set_y)
        Test['MTFL'] = MTFL_test

    if mask_300W:
        train_set_x_300W, train_set_y_300W = Train['300W']
        valid_set_x_300W, valid_set_y_300W = Valid['300W']
        W300_test = Test['300W']
        lfpw_set_x, lfpw_set_y = W300_test['lfpw']
        ibug_set_x, ibug_set_y = W300_test['ibug']
        Helen_set_x, Helen_set_y = W300_test['Helen']
        # 300W set process_once
        print "dist_ratio for 300W set is %f" %dist_ratio
        print "\ndoing preprocess_once on 300W train_set"
        train_set_x_300W, train_set_y_300W = preprocess_once(train_set_x_300W, train_set_y_300W, gray_scale=gray_scale, dist_ratio=dist_ratio)
        print "\ndoing preprocess_once on 300W valid_set"
        valid_set_x_300W, valid_set_y_300W = preprocess_once(valid_set_x_300W, valid_set_y_300W, gray_scale=gray_scale, dist_ratio=dist_ratio)
        print "\ndoing preprocess_once on 300W Helen test_set"
        Helen_set_x, Helen_set_y = preprocess_once(Helen_set_x, Helen_set_y, gray_scale=gray_scale, dist_ratio=dist_ratio)
        print "\ndoing preprocess_once on 300W ibug test_set"
        ibug_set_x, ibug_set_y = preprocess_once(ibug_set_x, ibug_set_y, gray_scale=gray_scale, dist_ratio=dist_ratio)
        print "\ndoing preprocess_once on 300W lfpw test_set"
        lfpw_set_x, lfpw_set_y = preprocess_once(lfpw_set_x, lfpw_set_y, gray_scale=gray_scale, dist_ratio=dist_ratio)

        # 300W set process_iter for valid and test sets
        print "\ndoing preprocess_iter on 300W valid_set"
        valid_set_x_300W, valid_set_y_300W = preprocess_iter(valid_set_x_300W, valid_set_y_300W, rng_seed, target_dim=td,
                                                             jitter=True, scale_mul=scale_mul, translate_mul=translate_mul,
                                                             sanity=False, use_lcn=use_lcn, dset='300W', block_img=block_img,
                                                             fixed_block=fixed_block, rotation=rotation)
        print "\ndoing preprocess_iter on 300W Helen test_set"
        Helen_set_x, Helen_set_y = preprocess_iter(Helen_set_x, Helen_set_y, rng_seed, target_dim=td, jitter=False,
                                                   scale_mul=0., translate_mul=0., sanity=False, use_lcn=use_lcn, dset='300W', block_img=False)
        print "\ndoing preprocess_iter on 300W ibug test_set"
        ibug_set_x, ibug_set_y = preprocess_iter(ibug_set_x, ibug_set_y, rng_seed, target_dim=td, jitter=False,
                                                 scale_mul=0., translate_mul=0., sanity=False, use_lcn=use_lcn, dset='300W',
                                                 block_img=False)
        print "\ndoing preprocess_iter on 300W lfpw test_set"
        lfpw_set_x, lfpw_set_y = preprocess_iter(lfpw_set_x, lfpw_set_y, rng_seed, target_dim=td, jitter=False, scale_mul=0., translate_mul=0.,
                                                 sanity=False, use_lcn=use_lcn, dset='300W', block_img=False)

        # adding data to the sets
        Train['300W'] = (train_set_x_300W, train_set_y_300W)
        Valid['300W'] = (valid_set_x_300W, valid_set_y_300W)
        W300_test = OrderedDict()
        W300_test['ibug'] = (ibug_set_x, ibug_set_y)
        W300_test['lfpw'] = (lfpw_set_x, lfpw_set_y)
        W300_test['Helen'] = (Helen_set_x, Helen_set_y)
        Test['300W'] = W300_test

    sets = [Train, Valid, Test]
    return sets

def producer_process(process_ID, data_queue, seed_queue, data_sets, target_dim,
                     jitter, scale_mul, translate_mul, sanity, use_lcn, masks,
                     block_img, fixed_block, rotation):
    if '300W' in masks:
        train_set_x_300W, train_set_y_300W = data_sets['300W']
        data_queue_300W = data_queue['300W']
    if 'MTFL' in masks:
        train_set_x_MTFL, train_set_y_MTFL = data_sets['MTFL']
        data_queue_MTFL = data_queue['MTFL']

    produce_start_time = time.time()
    while True:
        dset, SEED = seed_queue.get()

        if not isinstance(SEED, EOF):
            #sys.stderr.write("process_ID: %i, jittering for seed %i\n" %(process_ID, SEED))
            start_time = time.time()
            # jittering the train_set for this epoch

            seed_rng = np.random.RandomState(SEED)

            if dset == 'MTFL':

                new_train_set_x, new_train_set_y = preprocess_iter(set_x=train_set_x_MTFL, set_y=train_set_y_MTFL,
                                                                   seed_rng=seed_rng, target_dim=target_dim, jitter=jitter,
                                                                   scale_mul=scale_mul, translate_mul=translate_mul, sanity=sanity,
                                                                   use_lcn=use_lcn, dset=dset, block_img=block_img,
                                                                   fixed_block=fixed_block, rotation=rotation)
                elem = [new_train_set_x, new_train_set_y]
                # the size of elem (new_train_set_x, new_train_set_y combined) for MTFL is about 56 Megabyte together
                data_queue_MTFL.put(SEED,elem)
                #sys.stderr.write('process_ID: %i, added SEED %i for set MTFL. Queue size is now %s\n' % (process_ID, SEED, data_queue_MTFL.qsize()))
            elif dset == '300W':
                new_train_set_x, new_train_set_y = preprocess_iter(set_x=train_set_x_300W, set_y=train_set_y_300W,
                                                                   seed_rng=seed_rng, target_dim=target_dim, jitter=jitter,
                                                                   scale_mul=scale_mul, translate_mul=translate_mul, sanity=sanity,
                                                                   use_lcn=use_lcn, dset=dset, block_img=block_img,
                                                                   fixed_block=fixed_block, rotation=rotation)
                elem = [new_train_set_x, new_train_set_y]
                # the size of elem (new_train_set_x, new_train_set_y combined) for 300W is about 21 Megabyte together
                data_queue_300W.put(SEED,elem)
                #sys.stderr.write('process_ID: %i, added SEED %i for set 300W. Queue size is now %s\n' % (process_ID, SEED, data_queue_300W.qsize()))
            else:
                raise Exception('cannot jitter dset %s' %(dset,))

            end_time = time.time() # the end of training time
            training_time = (end_time - start_time)
            #sys.stderr.write('process_ID: %i, jittering took %f minutes\n' % (process_ID, training_time / 60.))

        else:
            sys.stderr.write('process_ID: %i. Done with producing.\n' %(process_ID))
            produce_end_time = time.time()
            producing_time = (produce_end_time - produce_start_time)
            sys.stderr.write('jittring all data took %f minutes\n' % (producing_time / 60.))
            break

def create_producers(Train, mask_MTFL, mask_300W, td, scale_mul,
                     translate_mul, use_lcn, block_img, fixed_block,
                     rotation, num_epochs, num_procs):
    data_sets = OrderedDict()
    data_queue = OrderedDict()
    seed_queue = Queue()
    NUMBER_OF_PROCESSES = num_procs
    masks = []
    producers = None

    if mask_MTFL:
        masks.append('MTFL')
        # creating queue for the jittered data.
        train_set_x_MTFL, train_set_y_MTFL = Train['MTFL']
        data_sets['MTFL'] = (train_set_x_MTFL, train_set_y_MTFL)
        data_queue_MTFL = OrderedQueue(maxsize=6)
        data_queue['MTFL'] = data_queue_MTFL
        # initializing the seed_queue
        num_queue_elem = np.min((num_epochs, NUMBER_OF_PROCESSES + 1))
        for i in xrange(num_queue_elem):
            seed_queue.put(('MTFL', i))

    if mask_300W:
        masks.append('300W')
        # creating queue for the jittered data.
        train_set_x_300W, train_set_y_300W = Train['300W']
        data_sets['300W'] = (train_set_x_300W, train_set_y_300W)
        data_queue_300W = OrderedQueue(maxsize=15)
        data_queue['300W'] = data_queue_300W
        num_queue_elem = np.min((num_epochs, NUMBER_OF_PROCESSES + 9))
        for i in xrange(num_queue_elem):
            seed_queue.put(('300W', i))

    assert len(data_sets) > 0
    assert len(data_queue) > 0
    assert len(data_queue) == len(data_sets) == len(masks)

    sys.stderr.write("\nstarting %d workers\n" % NUMBER_OF_PROCESSES)

    producers = [Process(target=producer_process, args=(i, data_queue, seed_queue,
                 data_sets, td, True, scale_mul, translate_mul, False, use_lcn, masks, block_img,
                 fixed_block, rotation))
                 for i in xrange(NUMBER_OF_PROCESSES)]

    for pr in producers:
        pr.daemon = True
        pr.start()
    sys.stderr.write("\n\ndone with creating and starting workers.\n\n")
    return [producers, data_queue, seed_queue, NUMBER_OF_PROCESSES, num_queue_elem]


def load_preproc_initProcs(mask_MTFL, mask_300W, all_train, scale_mul, translate_mul,
                           gray_scale, use_lcn, dist_ratio, target_dim, block_img, fixed_block,
                           rotation, num_epochs, num_procs):

    start_time = time.time()
    td = (target_dim, target_dim)
    print "the target_dim as the model's input is %s" %(td,)

    # loading the data #
    sets = get_data(mask_MTFL=mask_MTFL, mask_300W=mask_300W,
                    all_train=all_train)

    # preprocessing the data
    sets = preprocess_data(sets=sets, mask_MTFL=mask_MTFL, mask_300W=mask_300W,
                           scale_mul=scale_mul, translate_mul=translate_mul, gray_scale=gray_scale, use_lcn=use_lcn,
                           dist_ratio=dist_ratio, target_dim=target_dim, block_img=block_img, fixed_block=fixed_block,
                           rotation=rotation)
    Train, Valid, Test = sets

    # starting processes
    producers, data_queue, seed_queue, NUMBER_OF_PROCESSES, num_queue_elem = create_producers(Train=Train, mask_MTFL=mask_MTFL, mask_300W=mask_300W,
                                                                                              td=td, scale_mul=scale_mul, translate_mul=translate_mul,
                                                                                              use_lcn=use_lcn, block_img=block_img, fixed_block=fixed_block,
                                                                                              rotation=rotation, num_epochs=num_epochs,
                                                                                              num_procs=num_procs)
    return [sets, producers, data_queue, seed_queue, NUMBER_OF_PROCESSES, start_time, num_queue_elem]


def train_convNet(nkerns, num_epochs, learning_rate, batch_size, sliding_window_lenght, task_stop_threshold, L2_coef,
                  L2_coef_out, L2_coef_ful, use_ada_delta, decay, param_path, train_cost, gray_scale, scale_mul,
                  translate_mul, param_seed, Lambda_coefs, file_suffix, mask_MTFL, mask_300W, use_lcn, dist_ratio,
                  sw_lenght, all_train, paral_conv, target_dim, bilinear, coarse_conv_size, only_fine_tune_struc,
                  coarse_mask_branch, block_img, fixed_block, bch_norm, dropout, rotation, denoise_conv,
                  param_path_cfNet, nMaps_shuffled, use_res_2, use_res_1, extra_fine, large_F_filter,
                  load_no_output_params, save_no_output_params, train_all_kpts, dropout_kpts, num_model_kpts,
                  conv_size, param_path_strucNet, weight_per_pixel, no_fine_tune_model, conv_per_kpt,
                  linear_conv_per_kpt, only_49_kpts, concat_pool_locations, zero_non_pooled, num_procs,
                  learn_upsampling):

    sets, producers, data_queue, seed_queue, NUMBER_OF_PROCESSES, start_time, num_queue_elem = load_preproc_initProcs(
                                    mask_MTFL=mask_MTFL, mask_300W=mask_300W, all_train=all_train,
                                    scale_mul=scale_mul, translate_mul=translate_mul, use_lcn=use_lcn,
                                    dist_ratio=dist_ratio, target_dim=target_dim, block_img=block_img,
                                    fixed_block=fixed_block, rotation=rotation, num_epochs=num_epochs,
                                    gray_scale=gray_scale, num_procs=num_procs)

    if gray_scale:
        num_img_channels = 1
    else:
        num_img_channels = 3

    parallel_start_time = time.time()

    message = ''

    if paral_conv:
        if paral_conv == 1.0:
            sys.stderr.write('training SumNet_MTFL\n')
            from RCN.models.SumNet_MTFL import Train
            tr = Train(data_queue=data_queue, seed_queue=seed_queue, nkerns=nkerns, num_epochs=num_epochs, learning_rate=learning_rate,
                       batch_size=batch_size, sliding_window_lenght=sliding_window_lenght, task_stop_threshold=task_stop_threshold,
                       L2_coef_common=L2_coef, L2_coef_branch=L2_coef_ful, use_ada_delta=use_ada_delta, decay=decay, param_path=param_path,
                       train_cost=train_cost, file_suffix=file_suffix, num_img_channels=num_img_channels, sets=sets, param_seed=param_seed,
                       num_procs=NUMBER_OF_PROCESSES, Lambda_coefs=Lambda_coefs, mask_MTFL=mask_MTFL, mask_300W=mask_300W, use_lcn=use_lcn,
                       producers=producers, sw_lenght=sw_lenght, target_dim=target_dim, bilinear=bilinear, coarse_mask_branch=coarse_mask_branch,
                       L2_coef_out=L2_coef_out, coarse_conv_size=coarse_conv_size, weight_per_pixel=weight_per_pixel, use_res_2=use_res_2,
                       conv_per_kpt=conv_per_kpt, linear_conv_per_kpt=linear_conv_per_kpt)
        elif paral_conv == 2.0:
            sys.stderr.write('training SumNet_300W\n')
            from RCN.models.SumNet_300W import Train
            tr = Train(data_queue=data_queue, seed_queue=seed_queue, nkerns=nkerns, num_epochs=num_epochs, learning_rate=learning_rate,
                       batch_size=batch_size, sliding_window_lenght=sliding_window_lenght, task_stop_threshold=task_stop_threshold,
                       L2_coef_common=L2_coef, L2_coef_branch=L2_coef_ful, use_ada_delta=use_ada_delta, decay=decay, param_path=param_path,
                       train_cost=train_cost, file_suffix=file_suffix, num_img_channels=num_img_channels, sets=sets, param_seed=param_seed,
                       num_procs=NUMBER_OF_PROCESSES, Lambda_coefs=Lambda_coefs, mask_MTFL=mask_MTFL, mask_300W=mask_300W, use_lcn=use_lcn,
                       producers=producers, sw_lenght=sw_lenght, target_dim=target_dim, bilinear=bilinear, bch_norm=bch_norm, dropout=dropout,
                       num_queue_elem=num_queue_elem, use_res_2=use_res_2, extra_fine=extra_fine, load_no_output_params=load_no_output_params,
                       coarse_conv_size=coarse_conv_size, conv_per_kpt=conv_per_kpt, linear_conv_per_kpt=linear_conv_per_kpt,
                       weight_per_pixel=weight_per_pixel)
        elif paral_conv == 3.0:
            sys.stderr.write('training RCN_MTFL\n')
            from RCN.models.RCN_MTFL import Train
            tr = Train(data_queue=data_queue, seed_queue=seed_queue, nkerns=nkerns, num_epochs=num_epochs, learning_rate=learning_rate,
                       batch_size=batch_size, sliding_window_lenght=sliding_window_lenght, task_stop_threshold=task_stop_threshold,
                       L2_coef_common=L2_coef, L2_coef_branch=L2_coef_ful, use_ada_delta=use_ada_delta, decay=decay, param_path=param_path,
                       train_cost=train_cost, file_suffix=file_suffix, num_img_channels=num_img_channels, sets=sets, param_seed=param_seed,
                       num_procs=NUMBER_OF_PROCESSES, Lambda_coefs=Lambda_coefs, mask_MTFL=mask_MTFL, mask_300W=mask_300W, use_lcn=use_lcn,
                       producers=producers, sw_lenght=sw_lenght, target_dim=target_dim, bilinear=bilinear, bch_norm=bch_norm,
                       dropout=dropout, num_queue_elem=num_queue_elem, extra_fine=extra_fine, save_no_output_params=save_no_output_params,
                       coarse_conv_size=coarse_conv_size, use_res_2=use_res_2, use_res_1=use_res_1)
        elif paral_conv == 4.0:
            sys.stderr.write('training RCN_MTFL_skip\n')
            from RCN.models.RCN_MTFL_skip import Train
            tr = Train(data_queue=data_queue, seed_queue=seed_queue, nkerns=nkerns, num_epochs=num_epochs, learning_rate=learning_rate,
                       batch_size=batch_size, sliding_window_lenght=sliding_window_lenght, task_stop_threshold=task_stop_threshold,
                       L2_coef_common=L2_coef, L2_coef_branch=L2_coef_ful, use_ada_delta=use_ada_delta, decay=decay, param_path=param_path,
                       train_cost=train_cost, file_suffix=file_suffix, num_img_channels=num_img_channels, sets=sets, param_seed=param_seed,
                       num_procs=NUMBER_OF_PROCESSES, Lambda_coefs=Lambda_coefs, mask_MTFL=mask_MTFL, mask_300W=mask_300W, use_lcn=use_lcn,
                       producers=producers, sw_lenght=sw_lenght, target_dim=target_dim, bilinear=bilinear, bch_norm=bch_norm, dropout=dropout,
                       num_queue_elem=num_queue_elem, extra_fine=extra_fine, save_no_output_params=save_no_output_params,
                       coarse_conv_size=coarse_conv_size, use_res_2=use_res_2, use_res_1=use_res_1)
        elif paral_conv == 5.0:
            sys.stderr.write('training RCN_300W\n')
            from RCN.models.RCN_300W import Train
            tr = Train(data_queue=data_queue, seed_queue=seed_queue, nkerns=nkerns, num_epochs=num_epochs, learning_rate=learning_rate,
                       batch_size=batch_size, sliding_window_lenght=sliding_window_lenght, task_stop_threshold=task_stop_threshold,
                       L2_coef_common=L2_coef, L2_coef_branch=L2_coef_ful, use_ada_delta=use_ada_delta, decay=decay, param_path=param_path,
                       train_cost=train_cost, file_suffix=file_suffix, num_img_channels=num_img_channels, sets=sets, param_seed=param_seed,
                       num_procs=NUMBER_OF_PROCESSES, Lambda_coefs=Lambda_coefs, mask_MTFL=mask_MTFL, mask_300W=mask_300W, use_lcn=use_lcn,
                       producers=producers, sw_lenght=sw_lenght, target_dim=target_dim, bilinear=bilinear, bch_norm=bch_norm, dropout=dropout,
                       num_queue_elem=num_queue_elem, use_res_2=use_res_2, use_res_1=use_res_1, extra_fine=extra_fine, large_F_filter=large_F_filter,
                       load_no_output_params=load_no_output_params, only_49_kpts=only_49_kpts, concat_pool_locations=concat_pool_locations,
                       zero_non_pooled=zero_non_pooled, learn_upsampling=learn_upsampling)
        elif paral_conv == 6.0:
            sys.stderr.write('training RCN_300W_skip\n')
            from RCN.models.RCN_300W_skip import Train
            tr = Train(data_queue=data_queue, seed_queue=seed_queue, nkerns=nkerns, num_epochs=num_epochs, learning_rate=learning_rate,
                       batch_size=batch_size, sliding_window_lenght=sliding_window_lenght, task_stop_threshold=task_stop_threshold,
                       L2_coef_common=L2_coef, L2_coef_branch=L2_coef_ful, use_ada_delta=use_ada_delta, decay=decay, param_path=param_path,
                       train_cost=train_cost, file_suffix=file_suffix, num_img_channels=num_img_channels, sets=sets, param_seed=param_seed,
                       num_procs=NUMBER_OF_PROCESSES, Lambda_coefs=Lambda_coefs, mask_MTFL=mask_MTFL, mask_300W=mask_300W, use_lcn=use_lcn,
                       producers=producers, sw_lenght=sw_lenght, target_dim=target_dim, bilinear=bilinear, bch_norm=bch_norm, dropout=dropout,
                       num_queue_elem=num_queue_elem)

    elif denoise_conv:
        if denoise_conv == 1.0:
            sys.stderr.write('training denoising_300W model\n')
            from RCN.models.Denoising_300W import Train
            tr = Train(data_queue=data_queue, seed_queue=seed_queue, nkerns=nkerns, num_epochs=num_epochs, batch_size=batch_size,
                       L2_coef=L2_coef, param_path=param_path, file_suffix=file_suffix, sets=sets, param_seed=param_seed,
                       num_procs=NUMBER_OF_PROCESSES, mask_MTFL=mask_MTFL, mask_300W=mask_300W, nMaps_shuffled=nMaps_shuffled,
                       producers=producers, sw_lenght=sw_lenght, target_dim=target_dim, num_queue_elem=num_queue_elem,
                       param_path_cfNet=param_path_cfNet, decay=decay, train_all_kpts=train_all_kpts, dropout_kpts=dropout_kpts,
                       num_model_kpts=num_model_kpts, conv_size=conv_size)
        elif denoise_conv == 2.0:
            sys.stderr.write('training denoising convnet model\n')
            from RCN.models.fine_tune_cfNet_structured import Train
            tr = Train(data_queue=data_queue, seed_queue=seed_queue, nkerns=nkerns, num_epochs=num_epochs, batch_size=batch_size,
                       L2_coef=L2_coef, param_path=param_path, file_suffix=file_suffix, sets=sets, param_seed=param_seed,
                       num_procs=NUMBER_OF_PROCESSES, mask_MTFL=mask_MTFL, mask_300W=mask_300W, producers=producers,
                       sw_lenght=sw_lenght, target_dim=target_dim, num_queue_elem=num_queue_elem, use_lcn=use_lcn,
                       param_path_cfNet=param_path_cfNet, param_path_strucNet=param_path_strucNet, decay=decay,
                       train_all_kpts=train_all_kpts, dropout_kpts=dropout_kpts, num_model_kpts=num_model_kpts,
                       conv_size=conv_size, num_img_channels=num_img_channels, no_fine_tune_model=no_fine_tune_model,
                       only_49_kpts=only_49_kpts, only_fine_tune_struc=only_fine_tune_struc)

    message = tr.train()

    for pr in producers:
        pr.join()
    if mask_300W:
        data_queue['300W'].close()
    if mask_MTFL:
        data_queue['MTFL'].close()

    end_time = time.time() # the end of training time
    training_time = (end_time - start_time)
    parallel_time = (end_time - parallel_start_time)
    sys.stderr.write('parallel processes took %f minutes\n' % (parallel_time / 60.))
    sys.stderr.write('running all program took %f minutes\n' % (training_time / 60.))
    return message


def get_nkerns(paral_conv=0.0, denoise_conv=0.0, **kwargs):
    SumNet_MTFL = [5, 16, 16, 32, 32, 48, 48, 48, 48, 48, 48]
    SumNet_300W = [68, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
    RCN_MTFL = [5, 16, 32, 48, 48, 48, 48, 32, 32, 16, 16, 48, 48, 48]
    RCN_300W = [68, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
    Denoising_300W = [68, 64, 64, 64]

    if paral_conv == 1:
        nkerns = SumNet_MTFL
    elif paral_conv == 2:
        nkerns = SumNet_300W
    elif paral_conv in [3, 4]:
        nkerns = RCN_MTFL
    elif paral_conv in [5, 6]:
        nkerns = RCN_300W

    if denoise_conv in [1, 2]:
            nkerns = Denoising_300W
    return nkerns

def get_target_dim(target_dim, bilinear):
    if bilinear and target_dim == 40:
        target_dim = 41
    if bilinear and target_dim == 80:
        target_dim = 81
    if not bilinear and target_dim == 41:
        target_dim = 40
    if not bilinear and target_dim == 81:
        target_dim = 80
    return target_dim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processing the convNet param list.')

    ################################################
    # getting the parameters from the command line #
    ################################################
    parser.add_argument('--num_epochs', type=int, default=-1)

    # the default learning_rate in TCDCN paper is 0.003
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sliding_window_lenght', type=int, default=200)

    # the default task_stop_threshold in TCDCN paper is 0.5
    parser.add_argument('--task_stop_threshold', type=float, default=0.5)
    # the default L2_coef in TCDCN paper is 1.0
    # L2 coef for all layers except the fully connected layer and the last layer
    parser.add_argument('--L2_coef', type=float, default=0.0001)

    # L2 coef for the fully-connected layer
    parser.add_argument('--L2_coef_ful', type=float, default=0.0001)

    # L2 coef for the output layer
    parser.add_argument('--L2_coef_out', type=float, default=None)

    # the lambda coefficient for the multi-tasking
    parser.add_argument('--Lambda', type=str, help='the coefficient for multi-tasking. If one value given, all tasks get\
                        the same multiplier, otherwise 4 values should be given seperated by comma with no space',  default='1.0')

    parser.add_argument('--use_ada_delta', action='store_false', default=True)
    # the decay coefficient for ada_delta, the default vaue is 0.95
    parser.add_argument('--decay', type=float, default=0.95)

    # complete pickle path to the param_file should be given should be given
    parser.add_argument('--param_path', type=str, default="")

    # complete pickle path to the param_file of first keypoint detection
    # convnet model
    parser.add_argument('--param_path_cfNet', type=str, default="")
    # complete pickle path to the param_file of the structured keypoint detection model
    parser.add_argument('--param_path_strucNet', type=str, default="")

    parser.add_argument('--nMaps_shuffled', type=int, help='the number of channels to be shuffled for each sample',default=35)

    # keypoint locations normalized by the inter-ocular distance
    parser.add_argument('--cost', choices=['cost_kpt', 'error_kpt_avg', 'cross_entropy'], default='cost_kpt')
    parser.add_argument('--gray_scale', action='store_false', default=True)
    parser.add_argument('--file_suffix', type=str, default="")

    # the multiplier of face_bounding box for jittering
    parser.add_argument('--dist_ratio', type=float, help='the multiplier of face_bounding box for jittering', default=1.3)

    # the jittering multiplier for train and valid sets
    parser.add_argument('--scale_mul', type=float, default=0.5)
    parser.add_argument('--translate_mul', type=float, default=0.5)

    # setting the seed for parameters initialization
    parser.add_argument('--param_seed', type=int, default=54621)

    parser.add_argument('--use_lcn', help='indicates whether to use lcn or not', action='store_true', default=False)

    # the lenght of the sliding window for value averaging
    parser.add_argument('--sw', type=int, default=-1)

    parser.add_argument('--denoise_conv', type=float, help='0.0=no_denoise, 1.0=denose_conv, 2.0=strcutred_kpt_dist\
                         3.0=mlp_denoise_model, 4.0=finetune_both_models, 5.0=DAE_with_RCN_maps, 6.0=fine_tune_cfNet_DAE\
                         7.0=multi_stage_DAE', choices=[0.0, 1.0, 2.0], default=0.0)

    parser.add_argument('--all_train', help='indicates whether to train on all trainset or not', action='store_true', default=False)

    parser.add_argument('--paral_conv', type=float, help='indicates whether to use the two branch parallel conv, multi-scale\
                         conv or none, choices=[1.0=SumNet_MTFL, 2=SumNet_300W, 3=RCN_MTFL,\
                         4=RCN_MTFL_skip, 5=RCN_300W, 6=RCN_300W_skip, 0=none]',
                         choices=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], default=0.0)

    parser.add_argument('--target_dim', type=int, help='dimension of input images to the model choices=[40, 41, 80, 81]', choices=[40, 80], default=80)

    parser.add_argument('--bilinear', help='indicates whether to use bilinear interpolation for upsampling or just repeating values ', action='store_true', default=False)

    parser.add_argument('--coarse_mask_branch', help='a string of four values (only 0 and 1) seperated by comma, starting from F going to S', type=str, default="1,1,1,1,1")

    parser.add_argument('--coarse_conv_size', type=int, help='conv_size of the coarset branch of conv models', default=3)

    parser.add_argument('--block_img', help='indicates whether to block part of the image as preprocessing or not', action='store_true', default=False)
    parser.add_argument('--fixed_block', help='indicates whether to randomly choose the block size or not, default is True', action='store_true', default=False)
    parser.add_argument('--bch_norm', help='indicates whether to use batch_normalization in convnet or not', action='store_true', default=False)
    parser.add_argument('--dropout', help='indicates whether to use dropout in convnet or not', action='store_true', default=False)

    parser.add_argument('--rotation', type=int, help='the max rotationation for preprocessing', default=40)
    parser.add_argument('--use_res_2', help='indicates whether to use 2by2 branch resolution or not', action='store_true', default=False)
    parser.add_argument('--use_res_1', help='indicates whether to use up to 1by1 branch resolution or not', action='store_true', default=False)
    parser.add_argument('--extra_fine', help='indicates whether to use extra fine layer in RCN_300W or not', action='store_true', default=False)
    parser.add_argument('--large_F_filter', help='indicates whether to use filter size of 5 for fine layers of RCN models or not',
                        action='store_true', default=False)
    parser.add_argument('--load_no_output_params', help='indicates whether to load the model\'s output layer params or not', action='store_true', default=False)
    parser.add_argument('--save_no_output_params', help='indicates whether to save the model\'s params excluding the output layer', action='store_true', default=False)
    parser.add_argument('--train_all_kpts', help='indicates to train on all keypoints when training denoise_conv model', action='store_true', default=False)
    parser.add_argument('--dropout_kpts', help='indicates to masks the random selected keypoint when training the structured model', action='store_true', default=False)

    parser.add_argument('--num_model_kpts', help='the number of cfNet keypoints to be used when building the structured model', type=int, default=68)
    parser.add_argument('--conv_size', type=int, help='conv_size of the strucutured kpt model', default=45)
    parser.add_argument('--weight_per_pixel', help='indicates whether to use a per-pixel weight (in each branch for each kpt)\
                           in SumNet model or not', action='store_true', default=False)
    parser.add_argument('--conv_per_kpt', help='indicates whether to use a convnet on all branch feature maps per keypoint\
                           in SumNet model or not', action='store_true', default=False)
    parser.add_argument('--linear_conv_per_kpt', help='indicates whether to use a convnet on all branch feature maps per keypoint\
                           in SumNet model or not, in this case the final convolution in each branch is linear', action='store_true', default=False)
    parser.add_argument('--no_fine_tune_model', help='indicates whether to fine_tune the parameters of the two models (CakeNet and denoising)\
                           in the joint model', action='store_true', default=False)
    parser.add_argument('--only_fine_tune_struc', help='if used indicates to only finetune the structured model', action='store_true', default=False)

    parser.add_argument('--only_49_kpts', help='indicates whether to evaluate only for 49 keypoints in fine_tune_cfNet_structured model or not\
                           ', action='store_true', default=False)

    parser.add_argument('--concat_pool_locations', help=('indicates whether to concatenate pooled locations (as zero-one tensor)'
                                                         ' or not in RCN model when merging branches'), action='store_true', default=False)
    parser.add_argument('--zero_non_pooled', help=('indicates whether to set to zero the non pooled values of the pre-pooled'
                                                   ' maps in RCN model when concatenating feature maps or not'), action='store_true', default=False)

    parser.add_argument('--num_procs', type=int, help='number_of_processors for jittering data', default=2)
    parser.add_argument('--learn_upsampling', help=('indicates whether to learn the weights for upsampling not')
                        , action='store_true', default=False)

    args = parser.parse_args()
    paral_conv = args.paral_conv
    print "paral_conv is %s" %(paral_conv,)

    ###########################
    # printing out the values #
    ###########################
    learning_rate = args.learning_rate
    print "learning_rate is %f" %(learning_rate)

    batch_size = args.batch_size
    print "batch_size is %i" %(batch_size)

    sliding_window_lenght = args.sliding_window_lenght
    print "sliding_window_lenght is %i" %(sliding_window_lenght)

    task_stop_threshold = args.task_stop_threshold
    print "task_stop_threshold is %f" %(task_stop_threshold)

    Lambda = args.Lambda
    Lambda = Lambda.split(',')
    if len(Lambda) == 1:
        lam = float(Lambda[0])
        Lambda_coefs = [lam, lam, lam, lam]
    else:
        assert len(Lambda) == 4
        Lambda = map(float, Lambda)
        Lambda_coefs = [Lambda[0], Lambda[1], Lambda[2], Lambda[3]]
    print "Lamda_coefs are %s" %(Lambda_coefs,)

    print "---------------------------------"

    param_seed = args.param_seed
    print "SEED for parameters initailization is: %i" %param_seed

    gray_scale = args.gray_scale
    print "gray_scale is %r" %(gray_scale)

    use_ada_delta = args.use_ada_delta
    # if learning rate is specified it should use sgd and not ada_delta
    if learning_rate != 0.003:
        use_ada_delta = False
    print "use_ada_delta is %r" %(use_ada_delta)

    all_train = args.all_train
    print "all_train is %r" %all_train

    param_path = args.param_path
    print "param_path for loading previously save params is %s" %(param_path)

    param_path_cfNet = args.param_path_cfNet
    print "param_path_cfNet for loading first convnet model is %s" %(param_path_cfNet)

    param_path_strucNet = args.param_path_strucNet
    print "param_path_strucNet for loading first convnet model is %s" %(param_path_strucNet)

    train_cost = args.cost
    print "cost is %s" %train_cost

    print "---------------------------------"

    ####################
    # important infor #
    ####################
    num_epochs = args.num_epochs
    # if num_epochs is given use the given value
    # else set it based on the dataset
    if num_epochs == -1:
        if args.mask_MTFL:
            num_epochs = 3000
        else:
            num_epochs = 10000
    print "num_epochs is %i" %(num_epochs)

    L2_coef = args.L2_coef
    print "L2_coef is %f" %(L2_coef)

    L2_coef_ful = args.L2_coef_ful
    print "L2_coef for the fully connected layer is %f" %(L2_coef_ful)

    L2_coef_out = args.L2_coef_out
    if L2_coef_out is None:
        L2_coef_out = L2_coef
    print "L2_coef for the output layer is %f" %(L2_coef_out)

    decay = args.decay
    print "the decay coef for adadelta is %f" %decay

    scale_mul = args.scale_mul
    print "scale_mul is %f" %scale_mul

    translate_mul = args.translate_mul
    print "translate_mul is %f" %translate_mul

    denoise_conv = args.denoise_conv
    print "denoise_conv is %s" %(denoise_conv,)

    if paral_conv in [2, 5, 6] or denoise_conv in [1, 2]:
        mask_MTFL = 0
        mask_300W = 1
    elif paral_conv in [1, 3, 4]:
        mask_300W = 0
        mask_MTFL = 1
    print "mask_MTFL is %f " %mask_MTFL
    print "mask_300W is %f " %mask_300W

    use_lcn = args.use_lcn
    print "use_lcn is %r " %use_lcn

    file_suffix = '_' + args.file_suffix
    print "file_suffix is %s" %(file_suffix)

    dist_ratio = args.dist_ratio
    print "dist_ratio is %f" %dist_ratio

    # if sw is given, use the given value
    # else set it to 10% of the num_epochs
    sw = args.sw
    if sw == -1:
        sw = num_epochs / 10
    print "The sw_lenght is %s" %(sw,)

    br_mask = args.coarse_mask_branch
    br_mask = br_mask.split(',')
    br_mask = map(string.strip, br_mask)
    br_mask = map(int, br_mask)
    assert all(x==0 or x==1 for x in br_mask)
    coarse_mask_branch = np.array(br_mask)
    print "The coarse mask_branch in SumNet_MTFL model is %s" %(coarse_mask_branch,)

    bilinear = args.bilinear
    print "bilinear is %s" %(bilinear, )

    target_dim = args.target_dim
    target_dim = get_target_dim(target_dim, bilinear)

    print "target_dim is %r" %(target_dim,)

    nkerns = get_nkerns(paral_conv, denoise_conv)
    print "nkerns are %s" %(nkerns,)

    coarse_conv_size = args.coarse_conv_size
    print "coarse_conv_size is %s" %(coarse_conv_size,)

    block_img = args.block_img
    print "block_img is %s" %(block_img,)

    fixed_block = args.fixed_block
    print "fixed_block is %s" %(fixed_block,)

    bch_norm = args.bch_norm
    print "bch_norm is %s" %(bch_norm,)

    dropout = args.dropout
    print "dropout is %s" %(dropout)

    rotation = args.rotation
    print "rotation is %s" %(rotation)

    nMaps_shuffled = args.nMaps_shuffled
    print "nMaps_shuffled is %s" %(nMaps_shuffled,)

    use_res_2 = args.use_res_2
    print "use_res_2 is %s" %(use_res_2)

    use_res_1 = args.use_res_1
    print "use_res_1 is %s" %(use_res_1)

    extra_fine = args.extra_fine
    print "extra_fine layer for 300W model is %s" %(extra_fine)

    large_F_filter = args.large_F_filter
    print "large_F_filter for 300W model is %s" %(large_F_filter)

    load_no_output_params = args.load_no_output_params
    print "load_no_output_params for 300W model is %s" %(load_no_output_params)

    save_no_output_params = args.save_no_output_params
    print "save_no_output_params is %s" %(save_no_output_params)

    train_all_kpts = args.train_all_kpts
    print "train_all_kpts is %s" %(train_all_kpts)

    dropout_kpts = args.dropout_kpts
    print "dropout_kpts is %s " %(dropout_kpts)

    num_model_kpts = args.num_model_kpts
    print "num_model_kpts is %s" %(num_model_kpts,)

    conv_size = args.conv_size
    print "conv_size is %s" %(conv_size,)

    weight_per_pixel = args.weight_per_pixel
    print "weight_per_pixel is %s" %(weight_per_pixel)

    conv_per_kpt = args.conv_per_kpt
    print "conv_per_kpt is %s" %(conv_per_kpt)

    linear_conv_per_kpt = args.linear_conv_per_kpt
    print "linear_conv_per_kpt is %s" %(linear_conv_per_kpt)

    no_fine_tune_model = args.no_fine_tune_model
    print "no_fine_tune_model is %s" %(no_fine_tune_model)

    only_fine_tune_struc = args.only_fine_tune_struc
    print "only_fine_tune_struc is %s" %(only_fine_tune_struc)

    only_49_kpts = args.only_49_kpts
    print "only_49_kpts is %s" %(only_49_kpts)

    concat_pool_locations = args.concat_pool_locations
    print "concat_pool_locations is %s" %(concat_pool_locations,)

    zero_non_pooled = args.zero_non_pooled
    print "zero_non_pooled is %s" %(zero_non_pooled,)

    num_procs = args.num_procs
    print "num_procs is %s" %(num_procs,)

    learn_upsampling = args.learn_upsampling
    print "learn_upsampling is %s" %(learn_upsampling)

    ####################################
    # saving the settings for this run #
    ####################################
    params = vars(args)
    if file_suffix != '_':
        params_name = 'shared_conv_setting%s.pickle' %file_suffix
    else:
        params_name = 'shared_conv_setting.pickle'
    params_path = dest_dir + '/' + params_name
    with open (params_path, 'wb') as fp:
        pickle.dump(params, fp)

    message = train_convNet(nkerns=nkerns, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size,
                            sliding_window_lenght=sliding_window_lenght, task_stop_threshold=task_stop_threshold,
                            L2_coef=L2_coef, L2_coef_out=L2_coef_out, L2_coef_ful=L2_coef_ful, use_ada_delta=use_ada_delta,
                            decay=decay, param_path=param_path, train_cost=train_cost, gray_scale=gray_scale,
                            scale_mul=scale_mul, translate_mul=translate_mul, param_seed=param_seed, Lambda_coefs=Lambda_coefs,
                            file_suffix=file_suffix, mask_MTFL=mask_MTFL, mask_300W=mask_300W, use_lcn=use_lcn,
                            dist_ratio=dist_ratio, sw_lenght=sw, all_train=all_train, paral_conv=paral_conv,
                            target_dim=target_dim, bilinear=bilinear, coarse_conv_size=coarse_conv_size, block_img=block_img,
                            coarse_mask_branch=coarse_mask_branch, fixed_block=fixed_block, bch_norm=bch_norm,
                            dropout=dropout, rotation=rotation, denoise_conv=denoise_conv, learn_upsampling=learn_upsampling,
                            param_path_cfNet=param_path_cfNet, nMaps_shuffled=nMaps_shuffled, only_49_kpts=only_49_kpts,
                            use_res_2=use_res_2, use_res_1=use_res_1, extra_fine=extra_fine, load_no_output_params=load_no_output_params,
                            large_F_filter=large_F_filter, save_no_output_params=save_no_output_params, train_all_kpts=train_all_kpts,
                            dropout_kpts=dropout_kpts, num_model_kpts=num_model_kpts, conv_size=conv_size, num_procs=num_procs,
                            param_path_strucNet=param_path_strucNet, weight_per_pixel=weight_per_pixel, conv_per_kpt=conv_per_kpt,
                            no_fine_tune_model=no_fine_tune_model, linear_conv_per_kpt=linear_conv_per_kpt,
                            concat_pool_locations=concat_pool_locations, zero_non_pooled=zero_non_pooled,
                            only_fine_tune_struc=only_fine_tune_struc)

    sys.stderr.write("file_suffix is %s\n" %(file_suffix ))
    sys.stderr.write("%s\n" %(message,))

