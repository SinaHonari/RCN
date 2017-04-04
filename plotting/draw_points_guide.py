from RCN.utils.convnet_tools import create_TCDCN_obejct

import numpy as np
import argparse
import cPickle as pickle
import cv2


def get_kpts(X, param_path):
    """
    This method shows how to use a trained model
    and predict keypoints on some random face images.
    This method is meant as a guideline on how to write
    such a script and might need to be tuned to work on any data.

    The method takes the path to the weights of a RCN Model and creates
    an RCN model and sets its weights to it. It then gets the RCN model's
    prediction on X.

    parameters:
    -----------
        X: 4D tensor of shape (#batch, #row=80, #col=80, #channels=1)
           X contains gray-scale face images of size 80 * 80

        param_path: the path to the trained model's parameters.

    returns:
    --------
        key-point predictions on images in X, which has the shape of
        (#batch, #kpts=68, 2).
    """
    # making an object of model
    tcdcn, params = create_TCDCN_obejct(param_path)

    # setting the conv params to the weights
    tcdcn.load_params(param_path)

    ###########################
    # getting kpt predictions #
    ###########################
    batch_size = 64
    num_batches = int(np.ceil(X.shape[0]/float(batch_size)))
    # X should be a 4D tensor of shape (#batch, #row=80, #col=80, #channels=1)
    assert X.ndim == 4
    assert X.shape[1] == X.shape[2] == 80
    # assert x_batch is grayscaled
    assert X.shape[-1] == 1
    n_kpts = 68

    kpt_pred = []
    for index in np.arange(num_batches):
        x_batch = X[index * batch_size: (index + 1) * batch_size]
        n_bch, _, _, _ = x_batch.shape
        bound_mask = np.zeros((n_bch, n_kpts))
        border_pixel = np.tile([0, 0, 79, 79], (n_bch, 1))

        kpt_pred_batch = tcdcn.get_keypoints_MTFL(x_batch, bound_mask,
                                                  border_pixel, dropout=0)
        kpt_pred.extend(kpt_pred_batch)

    kpt_pred = np.array(kpt_pred)
    batch, _ = kpt_pred.shape
    kpt_pred = kpt_pred.reshape(batch, n_kpts, 2)
    return kpt_pred


def plot_cross(img, kpt, color, lnt=1):
    kpt = map(int, kpt)
    x, y = kpt
    cv2.line(img=img, pt1=(x-lnt, y-lnt), pt2=(x+lnt, y+lnt), color=color)
    cv2.line(img=img, pt1=(x-lnt, y+lnt), pt2=(x+lnt, y-lnt), color=color)
    return img


def draw_kpts(img, kpts, color):
    for kpt in kpts:
        x_i = int(kpt[0])
        y_i = int(kpt[1])
        img = plot_cross(img, kpt=(x_i, y_i), color=color)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Getting keypoint prediction\
                                     using a trained model.')
    parser.add_argument('--path', type=str, help='the complete path to the\
                        model\'s parameter pickle file that starts with\
                        shared_conv_params', required=True)

    parser.add_argument('--img_path', type=str, help='the complete path to the\
                        pickle file that contains pre-processed images',
                        required=True)

    args = parser.parse_args()
    path = args.path
    img_path = args.img_path

    fp = open(img_path, 'r')
    imgs = pickle.load(fp)

    pred_kpts = get_kpts(imgs, path)

    n_bch = pred_kpts.shape[0]
    for img, kpts, id in zip(imgs, pred_kpts, np.arange(n_bch)):
        img_new = draw_kpts(img, kpts, color=(0, 255, 0))
        name = 'img_%s.png' % (id)
        cv2.imwrite(name, img_new)
