"""
This module reads job logs (the pickle files that start with 'epoch_log')
and gets the performance of different sets over a sliding window of a given lenght.
It then picks the sliding window with the minimum validation set error and selects the 
epoch with the minimum valiation set error in that sliding window and reports error on 
other sets in that epoch. Then, it sorts the results for all jobs in that directory for
every key and writes them to a file.

Note: by default the sliding window lenght is 10% of the training epochs
"""

import os
import numpy as np
import argparse
import cPickle as pickle
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Plotting charts for cost_kpt and error_kpt_avg for all pickle files in a folder.')
parser.add_argument('--dir', type=str, help='the complete path to the directory contraining the pickle log files', required=True)
parser.add_argument('--sw', type=int, help='sliding window size for number average', default=-1)
# optional inputs

args = parser.parse_args()
dir = args.dir
sw = args.sw

def get_min_epoch_using_sw(set, key, sw, set_name='valid_MTFL'):
    """
    finds the minimum sliding window for the validation set
    and returns the epoch with the minimum valid_set error
    in that sliding window.
    """
    sub_names = set_name.split('_')
    subset = set
    for sub in sub_names:
        subset = subset[sub]
    vals = subset[key]
    vals = np.array(vals)

    n_elems = len(vals)
    sw_vals = []
    for i in xrange(n_elems - sw + 1):
        sw_val = np.mean(vals[i : i + sw])
        sw_vals.append(sw_val)
    #import pdb; pdb.set_trace()
    # min_sw_start index is the starting epoch of the window in the original epoch_vals
    min_sw_start = np.argmin(sw_vals)
    min_sw_end = min_sw_start + sw
    min_epoch_index = np.argmin(vals[min_sw_start: min_sw_end])
    min_epoch = min_sw_start + min_epoch_index
    print "min_epoch is %i" %min_epoch
    return min_epoch

def get_min_valid_index(set, key, set_name='valid_MTFL', num_min_epochs=10):
    sub_names = set_name.split('_')
    subset = set
    for sub in sub_names:
        subset = subset[sub]
    vals = subset[key]
    vals = np.array(vals)
    #import pdb; pdb.set_trace()
    min_epochs = vals.argsort()[:num_min_epochs]
    return np.array(min_epochs)

def get_epoch_value(set, key, set_name, min_epochs):
    sub_names = set_name.split('_')
    subset = set
    for sub in sub_names:
        subset = subset[sub]
    vals = subset[key]
    vals = np.array(vals)
    min_epochs = np.array(min_epochs)
    #min = np.mean(vals[min_epochs])
    min = vals[min_epochs]
    return min

def change_list_to_nparray(orderedDict):
    for key in orderedDict.keys():
        orderedDict[key] = np.array(orderedDict[key])
    #return orderedDict

def get_keys(fl):
    set_names = []
    for key in fl.keys():
        if 'cost_kpt' in fl[key].keys():
            set_names.append(key)
        else:
            subfl = fl[key]
            for subkey in subfl.keys():
                if 'cost_kpt' in subfl[subkey].keys():
                    name = "%s_%s" %(key, subkey)
                    set_names.append(name)
                else:
                    subsubfl = subfl[subkey]
                    for subsubkey in subsubfl.keys():
                        if 'cost_kpt' in subsubfl[subsubkey].keys():
                            name = "%s_%s_%s" %(key, subkey, subsubkey)
                            set_names.append(name)
    return set_names

files = []
for file in os.listdir(dir):
    if file.startswith("epochs_log") and file.endswith(".pickle"):
        files.append(file)

subsets = OrderedDict()
hypers = []
suffix = []

for f in files:
    print "processing %s" %(f,)
    pkl_path = "%s/%s" %(dir, f)
    fl = pickle.load(open(pkl_path, "rb" ))
    hp = f[11:-7]
    while hp.startswith('-') or hp.startswith('_'):
        hp = hp[1:]
    hypers.append(hp)
    suffix.append(f)
    
    set_names = get_keys(fl)
    #import pdb; pdb.set_trace()
    if sw == -1:
        sw = len(fl['train']['error_kpt_avg'])/10

    # getting the validation sets
    val_sets = val_sets = [set for set in set_names if set.startswith('valid')]

    for val_set in val_sets:
        min_epochs = get_min_epoch_using_sw(set=fl, key='error_kpt_avg', sw=sw, set_name=val_set)
        for set_name in set_names:
            value = get_epoch_value(set=fl, key='error_kpt_avg', set_name=set_name, min_epochs=min_epochs)
            if set_name not in subsets.keys():
                subsets[set_name] = []
            subsets[set_name].append(value)
            
    print "done with %s" %(f,)

hypers = np.array(hypers) 
suffix = np.array(suffix) 

change_list_to_nparray(subsets)

out_path = "%s/sorted_results_sw_%i_validsets.txt" %(dir, sw)
out_file=open(out_path,'w')

clip_path = "%s/clipped_results_sw_%i_validsets.txt" %(dir, sw)
clip_file=open(clip_path,'w')

# clip threshold number of results
clip_threshold = 10

suffix_path = "%s/file_suffix_results_sw_%i_validsets.txt" %(dir, sw)
suffix_file=open(suffix_path,'w')

for key in subsets.keys():
        valset = subsets[key]
        sort_index = np.argsort(valset)
        out_file.write("\n\nSORTNG BASED ON PARAMETER: %s\n\n" %key)
        clip_file.write("\n\nSORTNG BASED ON PARAMETER: %s\n\n" %key)
        suffix_file.write("\n\nSORTNG BASED ON PARAMETER: %s\n\n" %key)
        isets = []
        for iname in subsets.keys():
                iset = subsets[iname]
                sorted_list = iset[sort_index]
                isets.append((iname, sorted_list))
        hyper_list = hypers[sort_index]
        suffix_list = suffix[sort_index]
        msg = ''
        for iname, iset in isets:
                msg += '%s, ' %(iname,)
        msg += " hyper_params\n\n"
        out_file.write(msg)
        clip_file.write(msg)
        suffix_file.write(msg)
        for index in xrange(len(hypers)):
                msg = ''
                for iname, iset in isets:
                        msg += '%s, ' %(iset[index],)
                msg += '%s\n\n' %(hyper_list[index],)
                out_file.write(msg)
                suffix_file.write("%s\n" %suffix_list[index])
                if index < clip_threshold:
                   clip_file.write(msg)

#getting average values over all runs
print "getting average values of all runs:"
out_file.write("\naverage values of all runs:\n\n")
clip_file.write("\naverage values of all runs:\n\n")
isets = []
for iname in subsets.keys():
        iset = subsets[iname]
        avg = np.mean(iset)
        isets.append((iname, avg))
hyper_list = hypers[sort_index]
suffix_list = suffix[sort_index]
msg = ''
for iname, avg in isets:
        msg += '%s, ' %(iname,)
msg += "\n\n"
for iname, avg in isets:
        msg += '%s, ' %(avg,)
msg += "\n"
out_file.write(msg)
clip_file.write(msg)

print "done"
