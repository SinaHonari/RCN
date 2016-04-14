# RCN
The code for Recombinator Networks paper.

# Installation

1 - Install theano by following the instruction given here:
http://deeplearning.net/software/theano/install.html

2 - Clone RCN repo

3 - In ~/.bashrc add the parent directory of the clonded RCN repo to PYTHONPATH: <br />
export PYTHONPATH=/path/to/parent/dir/of/RCN:$PYTHONPATH

############################################################

# Dataset Creation

There are two datasets: 5-keypoint and 68-keypoint datasets

## For 5-keypoint datasets:
Run create_raw_afw.py and create_raw_MTFL.py modules in preprocessing directory. <br />
create_raw_MTFL.py creates the train set and the AFLW test set.
create_raw_afw.py creates the AFW test set. <br />
Here is how to run these modules:

### Train and AFLW dataset creation (by create_raw_MTFL.py):
1 - Download the images from:
http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip

2 - Unzip the folder and pass the complete path to it to 'src_dir' when calling create_raw_MTFL.py module.

3 - Call create_raw_MTFL.py by passing complete path to 'src_dir' and 'dest_dir' arguments: <br />
python create_raw_MTFL.py  --src_dir=/complete/path/to/MTFL/unzipped/folder --dest_dir=/complete/path/to/RCN/datasets

Note: dest_dir is the location where the dataset will be created. It should be finally put in RCN/datasets directory
of the repo

This module will create MTFL_test_160by160.pickle and MTFL_train_160by160.pickle in the given dest_dir path.

### AFW dataset creation (by create_raw_afw.py):

1 - Download the images from:
https://www.ics.uci.edu/~xzhu/face/AFW.zip

2 - Unzip the folder and pass the complete path to it to 'src_dir' when calling create_raw_MTFL.py module.

3 - Call create_raw_MTFL.py module by passing complete path to 'src_dir' and 'dest_dir' arguments: <br />
python create_raw_afw.py  --src_dir=/complete/path/to/AFW/unzipped/folder --dest_dir=/complete/path/to/RCN/datasets

This module will create AFW_test_160by160.pickle in the given dest_dir path.

## For 68-keypoint datasets:
Run create_raw_300W.py module in preprocessing directory as follows:

1 - Download Helen, LFPW, AFW and IBUG datasets from:
http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

2 - Once unzipped, the helen and lfpw have two subdirectories, 'testset' and 'trainset'.
Rename them to 'X_testset' and 'X_trainset', for each dataset X.

3 - Create one directory named 'Train_set' and put unzipped 'afw', 'helen_trainset'
and 'lfpw_trainset' directories into it (as three sub-directories).

4 - Create another directory named 'Test_set' and put unzipped 'ibug', 'helen_testset' and 'lfpw_testset'
into it (as three sub-directories).

5 - Put 'Train_set' and 'Test_set' directories into one direcotory (i.e. 300W) and pass
the complete path to it to 'src_dir' when calling this module.

6 - Call create_raw_300W.py module by passing complete path to 'src_dir' and 'dest_dir' arguments: <br />
python create_raw_300W.py --src_dir=/complete/path/to/300W/folder --dest_dir=/complete/path/to/RCN/datasets

This module will create 300W_test_160by160.pickle and 300W_train_160by160.pickle files in the given dest_dir path.

########################################################

# Model Training
Create_procs.py module in RCN/models should be called with the right arguments to train each model. <br />
The Theano flag 'THEANO_FLAGS=floatX=float32,device=gpu,force_device=True' right before the python command
should be used to train the model on gpu and set the type of shared varibles to float32: <br />
THEANO_FLAGS=floatX=float32,device=gpu,force_device=True python create_procs.py <'flags'>

Here is the set of flags to be used for training each model:

### Training SumNet for MTFL:
python create_procs.py --L2_coef=1e-06 --L2_coef_ful=0.0001 --L2_coef_out=1e-14 --file_suffix=SumNet_MTFL_test --num_epochs=2  --paral_conv=1.0 --use_lcn --block_img

optional flags: --use_res_2, --weight_per_pixel

###########

### Training SumNet for 300W:
python create_procs.py --L2_coef=1e-08 --L2_coef_ful=1e-12 --file_suffix=SumNet_300W_test --num_epochs=2 --paral_conv=2.0 --use_lcn --block_img 

optional flags: --use_res_2, --weight_per_pixel

###########

### Training RCN for MTFL:

python create_procs.py --L2_coef=1e-12 --L2_coef_ful=1e-08 --file_suffix=RCN_MTFL_test --num_epochs=2 --paral_conv=3.0 --use_lcn --block_img 

optional flags: --use_res_2, --use_res_1

###########

### Training RCN for MTFL with skip connections:

python create_procs.py --L2_coef=0.1 --L2_coef_ful=0.1 --file_suffix=RCN_MTFL_skip_test --num_epochs=2  --paral_conv=4.0 --use_lcn --block_img

optional flags: --use_res_2, --use_res_1

###########

### Training RCN for 300W:

python create_procs.py --L2_coef=0.01 --L2_coef_ful=1e-12 --file_suffix=RCN_300W_test --num_epochs=2 --paral_conv=5.0 --use_lcn --block_img

optional flags: --use_res_2, --use_res_1

###########

### Training RCN for 300W with skip connections:

python create_procs.py --L2_coef=1e-10 --L2_coef_ful=1e-08 --file_suffix=RCN_300W_skip_test  --num_epochs=2 --paral_conv=6.0 --use_lcn --block_img

###########

### Training Denoising model for 300W:

python create_procs.py --L2_coef=1e-06 --file_suffix=Denoising_300W_test --num_epochs=2  --denoise_conv=1.0

OR 

python create_procs.py --L2_coef=1e-06 --file_suffix=Denoising_300W_cfNet_test --num_epochs=2  --denoise_conv=1.0 --param_path_cfNet=path/to/trained/RCN_300W/model/parameters

Note : the latter loads the parameters of a trained RCN model, generates one_hot maps and gives them to the 
denoising model when evaluating 300W test sets (not on the train or valid sets). <br />
optional flags: --nMaps_shuffled=35, --conv_size=45

### Optional flag information:
--use_res_2: indicates to go to resolution 2*2 as the coarsest resolution in the model. Default is 5*5. <br />
--use_res_1: indicates to go to resolution 1*1 as the coarsest resolution in the model. Default is 5*5. <br />
--weight_per_pixel: indicates in the SumNet model a weight per pixel to be used when summing 
the upsampled feature maps of different branches (default is one weight per feature map). <br />
--batch_size: the batch size for training model in each iteration. Default is 64. If the model is too big to be set on gpu, use smaller
values. <br />
--file_suffix: gives a name to the model being trained (You can differentiate multiple trained models using this flag). <br />
--nMaps_shuffled: the number of keypoints to be jittered when training the denoising model. Default is 35. <br />
--conv_size: the size of the convolutional kernel when training the denoising model. Default is 45. <br />
###########

### Output files:
Running one of the above models, e.g. with --file_suffix=test trained on MTFL dataset for 100 epochs,
generates the following 5 outputs in RCN/models/exp_shared_conv directory: <br />
adadelta_params_test.pickle -> keeps parameters of adadelta training algorithm <br />
shared_conv_params_epoch_100_test_MTFL.pickle -> keeps the parameters of the model in the final epoch <br />
shared_conv_params_test_MTFL.pickle -> keeps the parameters of the model based on best validation set performance <br />
shared_conv_setting_test.pickle -> keeps the flags used to run this model <br />
epochs_log_test.pickle -> keeps the logs of each epoch while training the model <br />

########################################################

# Citation
If you use this code please cite: <br />
Sina Honari, Jason Yosinski, Pascal Vincent, Christopher Pal, Recombinator Networks: Learning Coarse-to-Fine Feature Aggregation, in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

