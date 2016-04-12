import numpy as np
import theano.tensor as T

def gaussian(size, sigma):
    height = float(size)
    width = float(size)
    center_x = width  / 2. + 0.5
    center_y = height / 2. + 0.5

    gauss = np.zeros((int(height), int(width)))

    for i in xrange(1, int(height) + 1):
        for j in xrange(1, int(height) + 1):
            x_diff_sq = ((float(j) - center_x)/(sigma*width)) ** 2.
            y_diff_sq = ((float(i) - center_y)/(sigma*height)) ** 2.
            gauss[i-1][j-1] = np.exp( - (x_diff_sq + y_diff_sq) / 2.)

    return gauss


def lcn(x, nchannels, dim, size=9):
    """
    measures local contrast normalization for a batch
    """
    g = gaussian(size,1.591/size)
    g /= g.sum()
    g = np.float32(g.reshape((1, 1, size,size)))
    g = np.tile(g, (1, nchannels, 1, 1))
    mean = T.nnet.conv.conv2d(x, T.constant(g),
                              (None, nchannels, dim, dim),
                              (1, nchannels, size, size),
                              'full')
    mean = mean[:, :, size/2: dim+size/2,
                size/2: dim+size/2]
    v = x - mean
    var = T.nnet.conv.conv2d(T.sqr(v),T.constant(g),
                             (None, nchannels, dim, dim),
                             (1, nchannels, size, size),
                             'full')
    var = var[:, :, size/2: dim+size/2,
              size/2: dim+size/2]
    std = T.sqrt(var)
    out = v / T.maximum(std, T.mean(std))
    return (out + 2.5 )/5
