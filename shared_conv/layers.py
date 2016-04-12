import theano
import theano.tensor as T
import numpy as np
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d, max_pool_2d_same_size
from theano.tensor.nnet.abstract_conv import conv2d_grad_wrt_inputs
from facepoints.nade.nade_regression import ff_layer, nade_fprop_universal, nade_predict_universal
from facepoints.nade.nade_regression import orderless_nade_fprop_universal, orderless_nade_predict_universal
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def apply_act(x, act=None):
    # apply act(x)
    # linear:0, sigmoid:1, tanh:2, relu:3, softmax:4, ultra_fast_sigmoid:5
    if act == 'sigmoid' or act == 1:
        rval = T.nnet.sigmoid(x)
    elif act == 'tanh' or act == 2:
        rval = T.tanh(x)
    elif act == 'linear' or act == 0:
        rval = x
    elif act == 'softmax' or act == 4:
        rval = T.nnet.softmax(x)
    elif act == 'absTanh' or act == 5:
        rval = T.abs_(T.tanh(x))
    elif act == 'relu' or act == 6:
        relu = lambda x: x * (x > 1e-8)
        rval  = relu(x)
    elif act == 'lrelu' or act == 7:
        lrelu = lambda x: x * (x > 1e-8) + 0.01 * x * (x <= 1e-8)
        rval  = lrelu(x)
    return rval

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        Note:
        -------
        If you want to reduce the number of parameters, reduce one from n_out
        right at the beginning of the constructor, which makes the W and b have
        one less output dimension than the number of classes. Then instead of
        feeding 'softmax_input' to the softmax, add one column of zeros to end
        of it, and then feed it to the softmax. This will result in softmax
        prediction for as many as the number of classes.
        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)

        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='lgReg_W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='lgReg_b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k

        softmax_input = T.dot(input, self.W) + self.b
        self.p_y_given_x = T.nnet.softmax(softmax_input)
        #self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W))

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.neq(self.y_pred, y)
        else:
            raise NotImplementedError()

class Softmax(object):
    """Multi-class Softmax layer

    It's the same as Multi-class LogisticRegression with the exception that, it does not have
    any weights of it's own.
    """

    def __init__(self, input):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)
        """

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # input is a matrix where row-j  represents input training sample-j
        self.p_y_given_x = T.nnet.softmax(input)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]]
        # Note that the mean ecross mini-batch is taken outside of this method
        return -(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def predict(self, y):
        """for each example in the minibatch, returns the result of argmax
           prediction of the layer
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return self.y_pred
        else:
            raise NotImplementedError()

class ConvPoolLayer(object):
    """conv and Pool Layers of a convolutional network """

    def __init__(self, rng, input, image_shape, filter_shape, dropout, pool=True, poolsize=(2, 2),
                 border_mode='valid', act='absTanh', bch_norm=False, param_seed=54621,
                 use_params=False, conv_params=None):
        """
        Allocate a ConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type pool: boolean
        :param pool: indicates whether pooling should be used after convolution or not

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        if use_params:
            # use previously defined parameters
            self.W = conv_params[0]
            self.b = conv_params[1]
        else:
            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = np.prod(filter_shape[1:])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            # pooling size
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                       np.prod(poolsize))
            # initialize weights with random weights
            W_bound = np.sqrt(6. / (fan_in + fan_out))

            self.W = theano.shared(
                np.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                name='Conv_W',
                borrow=True
            )

            # the bias is a 1D tensor -- one bias per output feature map
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='Conv_b', borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode=border_mode
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height

        if bch_norm:
            np_g = np.ones(filter_shape[0]).astype(theano.config.floatX)
            g = theano.shared(np_g, name='conv_bn_g', borrow=True)
            self.g = g
            normed = (conv_out - conv_out.mean(axis=(0, 2, 3), keepdims=True)) / (conv_out.std(
                      axis=(0, 2, 3), keepdims=True) + 1E-6)
            pre_act = self.g.dimshuffle('x', 0, 'x', 'x') * normed + self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            pre_act = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        conv_func_out = apply_act(pre_act, act)

        # if pooling should be done, downsample the images
        if pool:
            # downsample each feature map individually, using maxpooling
            pooled_out = pool_2d(input=conv_func_out, ds=poolsize,
                                 ignore_border=True)
        else:
            pooled_out = conv_func_out

        # use dropout=1 in train and dropout=0 in test
        self.dropout = dropout
        srng = RandomStreams(seed=param_seed)
        pooled_out = T.switch(T.gt(self.dropout, 0), pooled_out * srng.normal(
                                        size=theano.tensor.shape(pooled_out),
                                        avg=1.0, std=1.0), pooled_out)
        self.output = pooled_out

        # store parameters of this layer
        if bch_norm:
            self.params = [self.W, self.b, self.g]
        else:
            self.params = [self.W, self.b]


class UpsamplingLayer(object):
    """upsampling layer using transpose convolution """

    def __init__(self, rng, output_grad, ratio):
        """
        This class turns the output_grad into a tensor of shape
        (#bch * #kernels, 1, #rows, #cols) and then applied the
        transpose convolution by using a kernel of shape
        (1, 1, filter_shape[0], filter_shape[1]). This will upsample
        each feature map independetly of the values in the other feature maps.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights.

        :type output_grad: theano.tensor.dtensor4
        :param output_grad: symbolic tensor that should be upsampled.

        :type ratio: int
        :param ratio: the amount by which output_grad will be upsampled.
        """
        # reshaping output_grad to have only one feature map
        bch, ch, row, col = output_grad.shape
        output_grad_reshaped = output_grad.reshape((-1, 1, row, col))

        up_conv_size = ratio * 2 - 1

        # input_shape is the shape of the upsampled tensor
        input_shape = (None, None, row * ratio, col * ratio)

        # filter shape is 2D kernel shape that will be applied
        # to upsample the feature maps
        filter_shape = (1, 1, up_conv_size, up_conv_size)

        border_mode = (ratio - 1, ratio - 1)
        subsample = (ratio, ratio)

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        # pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))

        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            name='Conv_W',
            borrow=True
        )

        # convolve input feature maps with filters
        conv_out = conv2d_grad_wrt_inputs(
            output_grad=output_grad_reshaped,
            filters=self.W,
            input_shape=input_shape,
            filter_shape=filter_shape,
            border_mode=border_mode,
            subsample=subsample,
            filter_flip=True
        )

        self.output = conv_out.reshape((bch, ch, row * ratio, col * ratio))
        self.params = [self.W]


class PoolLayer(object):
    """Pool Layer of a convolutional network """
    def __init__(self, input, dropout, poolsize=(2, 2), stride_size=None, ignore_border=True, param_seed=54621):
        """
        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)

        :type ignore_border: boolean
        :param ignore_border: indicates whether to ingnore_border when pooling or not
        """

        # downsample each feature map individually, using maxpooling
        pooled_out = pool_2d(input=input, ds=poolsize, st=stride_size,
                             ignore_border=ignore_border)

        # use dropout=1 in train and dropout=0 in test
        self.dropout = dropout
        srng = RandomStreams(seed=param_seed)
        pooled_out = T.switch(self.dropout > 0, pooled_out * srng.normal(
                                        size=theano.tensor.shape(pooled_out),
                                        avg=1.0, std=1.0), pooled_out)

        self.output = pooled_out

class PoolSameSizeLayer(object):
    """Pool Layer of a convolutional network """
    def __init__(self, input, patch_size=(2, 2)):
        """
        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)

        :type ignore_border: boolean
        :param ignore_border: indicates whether to ingnore_border when pooling or not
        """

        # downsample each feature map individually, using maxpooling
        pooled_out = max_pool_2d_same_size(input=input, patch_size=patch_size)
        self.output = pooled_out

class NadeLayer(object):
    def __init__(self, rng, input, output, nade_in, nade_out, hidden_act, output_act):
        """
        Nade layer.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.fmatrix
        :param input: a symbolic tensor of shape (n_examples, nade_in)

        :type output: theano.tensor.fmatrix
        :param output: a symbolic tensor of shape (n_examples, nade_out)

        :type nade_in: int
        :param nade_in: dimensionality of input

        :type nade_out: int
        :param nade_out: number of hidden units

        :type activation: string, one of ['sigmoid', 'tanh', 'linear', 'softmax', 'relu', 'absTanh']
        :param activation: Non linearity to be applied in the hidden layer
        """
        self.input = input

        self.w_nade, _ = ff_layer(rng, n_in=nade_out, n_out=nade_in)
        self.v_nade, self.b_nade = ff_layer(rng, n_in=nade_in, n_out=nade_out)

        # nade cost at train time
        nade_out = nade_fprop_universal(output, self.b_nade, input, self.w_nade, self.v_nade, hidden_act, output_act)
        self.output = nade_out

        # nade error at test time
        self.predict = nade_predict_universal(self.b_nade, input, self.w_nade, self.v_nade, hidden_act, output_act)

        # store parameters of this layer
        self.params = [self.w_nade, self.v_nade, self.b_nade]

class RNadeLayer(object):
    def __init__(self, rng, input, output, nade_in, nade_out, mask, ordering, hidden_act, output_act):
        """
        OrderLess Nade layer.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.fmatrix
        :param input: a symbolic tensor of shape (n_examples, nade_in)

        :type output: theano.tensor.fmatrix
        :param output: a symbolic tensor of shape (n_examples, nade_out)

        :type nade_in: int
        :param nade_in: dimensionality of input

        :type nade_out: int
        :param nade_out: number of hidden units

        :type mask: theano.tensor.fmatrix
        :param mask: a mask indicating which columns per sample(row) should be masked
        : Note: 0 values means use that column and 1 values means ignore that column

        :type ordering: theano.tensor.ivector
        :param ordering: the orders of the elemenent that nade should predict

        :type activation: string, one of ['sigmoid', 'tanh', 'linear', 'softmax', 'relu', 'absTanh']
        :param activation: Non linearity to be applied in the hidden layer
        """
        self.input = input

        self.w_nade, _ = ff_layer(rng, n_in=nade_out, n_out=nade_in)
        self.v_nade, self.b_nade = ff_layer(rng, n_in=nade_in, n_out=nade_out)
        self.Wflags, _ = ff_layer(rng, nade_out, nade_in)

        # nade cost at train time
        nade_out = orderless_nade_fprop_universal(output, self.b_nade, input, self.w_nade, self.v_nade, mask, self.Wflags, hidden_act, output_act)
        self.output = nade_out

        # nade error at test time
        self.predict = orderless_nade_predict_universal(self.b_nade, input, self.w_nade, self.v_nade, self.Wflags, ordering, hidden_act, output_act)

        # store parameters of this layer
        self.params = [self.w_nade, self.v_nade, self.b_nade, self.Wflags]

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, train_time, W=None, b=None,
                 act=None, dropout_prob=0.5, dropout=False):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type act: string
        :param act: Non linearity to be applied in the hidden layer

        :type dropout: bool
        :param dropout: a boolean parameter indicating whether or not
                            dropout is used for the HiddenLayer. It's indicated
                            once, while building the graph

        :type train_time: theano scalar
        :param train_time: a mask indicating whether it's train-time (1.0)
                           or test-time (0.0) to multiply apply dropout accordingly

        :type dropout_prob: float
        :param dropout_prob: the probability of dropout
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note: optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.

        W_bound = np.sqrt(6. / (n_in + n_out))

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-W_bound,
                    high=W_bound,
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if act == 'sigmoid':
                W_values *= 4.

            W = theano.shared(value=W_values, name='HidL_W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='Hid_b', borrow=True)

        self.W = W
        self.b = b

        self.dropout = dropout
        self.dropout_prob = dropout_prob
        if dropout:
            # theano.tensor.shared_randomstreams.RandomStreams creates a different
            # random value in each call to the graph. using rng.randint(999999)
            # also ensures that different Hidden layers don't get the same initialization
            srng = theano.tensor.shared_randomstreams.RandomStreams(
                        rng.randint(999999))

        if dropout:
            # at train time, train_time is one and only the first term is used, which uses the model's weights
            # at test time, train_time is zero and only the second term is used, which uses a ratio of the model's weights
            W_used = train_time * self.W + (1. - train_time) * self.W * (1. - self.dropout_prob)
        else:
            W_used = self.W

        lin_output = T.dot(input, W_used) + self.b

        output = (
            lin_output if act is None
            else apply_act(lin_output, act)
        )

        if dropout:
            # p=1-p because 1's indicate keep and p is prob of dropping
            mask = srng.binomial(n=1, p=1-dropout_prob, size=output.shape)
            # at train time, train_time is one and only the first term is used, which masks the output units
            # at test time, train_time is zero and only the second term is used, which uses all output units
            # The cast is important because
            # int * float32 = float64 which pulls things off the gpu
            self.output = train_time * output * T.cast(mask, theano.config.floatX) +\
                          (1. - train_time) * output
        else:
            self.output = output

        # parameters of the model
        self.params = [self.W, self.b]
