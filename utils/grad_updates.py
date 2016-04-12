from __future__ import division
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle as pkl
import numpy
import copy
import os, sys, socket, shutil
import time
from collections import OrderedDict
floatX='float32'

# some utilities
def constantX(value, float_dtype='float32'):
    return theano.tensor.constant(numpy.asarray(value, dtype=float_dtype))

def sharedX(value):
    return theano.shared(value)

class Train_alg(object):
    def build_updates(self,cost, params, consider_constant,
                      clip_c=0,clip_idx=None,
                      shrink_grad=None, choice=None, decay=0.95):
        updates = OrderedDict()
        grads = T.grad(cost, params, consider_constant)
        def apply_clip(g):
            g2 = 0.
            g2 += (g**2).sum()
            new_grad = T.switch(g2 > (clip_c**2),
                            g / T.sqrt(g2) * clip_c,
                            g)
            return new_grad
        if clip_c > 0. and clip_idx is not None:
            for idx in clip_idx:
                grads[idx] = apply_clip(grads[idx])
        if shrink_grad is not None:
            for idx in shrink_grad:
                grads[idx] *= 0.001

        def get_updates_adadelta(grads,params,decay=0.95):
            print 'build updates with adadelta'
            print 'decay value is %f' %decay
            decay = constantX(decay)
            self.params = []
            for param, grad in zip(params, grads):
                # mean_squared_grad := E[g^2]_{t-1}
                mean_square_grad = sharedX(numpy.zeros(param.get_value().shape, dtype=floatX))
                # mean_square_dx := E[(\Delta x)^2]_{t-1}
                mean_square_dx = sharedX(numpy.zeros(param.get_value().shape, dtype=floatX))
                if param.name is not None:
                    mean_square_grad.name = 'mean_square_grad_' + param.name
                    mean_square_dx.name = 'mean_square_dx_' + param.name

                # Accumulate gradient
                new_mean_squared_grad = \
                        decay * mean_square_grad +\
                        (1. - decay) * T.sqr(grad)
                # Compute update
                epsilon = constantX(1e-8)
                rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
                rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon)
                delta_x_t = - rms_dx_tm1 / rms_grad_t * grad

                # Accumulate updates
                new_mean_square_dx = \
                        decay * mean_square_dx + \
                        (1. - decay) * T.sqr(delta_x_t)

                # Apply update
                updates[mean_square_grad] = new_mean_squared_grad
                updates[mean_square_dx] = new_mean_square_dx
                updates[param] = param + delta_x_t
                self.params.extend([mean_square_grad, mean_square_dx])

        def get_updates_grads_momentum(gparams, params, lr=0.1, momentum=0.5):
            print 'building updates with momentum'
            # build momentum
            gparams_mom = []
            for param in params:
                gparam_mom = theano.shared(
                    numpy.zeros(param.get_value(borrow=True).shape,
                    dtype=floatX))
                gparams_mom.append(gparam_mom)

            self.params = gparams_mom
            for gparam, gparam_mom, param in zip(gparams, gparams_mom, params):
                inc = momentum * gparam_mom - (constantX(1) - momentum) * lr * gparam
                updates[gparam_mom] = inc
                updates[param] = param + inc

        def get_updates_rmsprop(grads, params, lr=0.1, decay=0.95):
            self.params = []
            for param,grad in zip(params,grads):
                mean_square_grad = sharedX(numpy.zeros(param.get_value().shape, dtype=floatX))
                new_mean_squared_grad = (decay * mean_square_grad +
                                         (1. - decay) * T.sqr(grad))
                rms_grad_t = T.sqrt(new_mean_squared_grad)
                delta_x_t = constantX(-1) * lr * grad / rms_grad_t
                updates[mean_square_grad] = new_mean_squared_grad
                updates[param] = param + delta_x_t
                self.params.append(mean_square_grad)
 
        get_updates_adadelta(grads, params, decay)
        #get_updates_grads_momentum(grads, params)
        #get_updates_rmsprop(grads, params)
        return updates
