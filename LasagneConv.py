'''The Lasagne NN module for Daya Bay data'''

import numpy as np
import theano
import theano.tensor as T
import lasagne as l

from Network import AbstractNetwork

class IBDPairConvAe(AbstractNetwork):
    '''A convolutional autoencoder for interpreting candidate IBD pairs.'''

    def __init__(self):
        '''Initialize a ready-to-train convolutional autoencoder.'''
        super(IBDPairConvAe, self).__init__(self)
        # Shapes are given as (batch, depth, height, width)
        self.minibatch_shape = (128, 4, 8, 24)
        self.minibatch_size = self.minibatch_shape[0]
        self.image_shape = self.minibatch_shape[1:-1]
        self.epochs = 1
        self.learn_rate = 0.0001
        self.bottleneck_width = 10
        self.input_var = T.dtensor4('input')
        self.network = self._setup_network()
        self.cost = self._setup_cost()
        self.optimizer = self._setup_optimizer()
        self.train_once = theano.function([self.input_var],
            self.cost, updates=self.optimizer)

    def _setup_network(self):
        '''Construct the ConvAe architecture for Daya Bay IBDs.'''
        initial_weights = l.init.Normal(1, 0)
        # Input layer shape = (minibatch_size, 4, 8, 24)
        network = l.layers.InputLayer(
            input_var=self.input_var,
            shape=self.minibatch_shape)
        # post-conv shape = (minibatch_size, 16, 8, 24)
        network = l.layers.Conv2DLayer(
            network,
            num_filters=16,
            filter_size=(5, 5),
            pad=(2, 2),
            W=initial_weights,
            nonlinearity=l.nonlinearities.rectify)
        # post-pool shape = (minibatch_size, 16, 4, 12)
        network = l.layers.MaxPool2DLayer(
            network,
            pool_size=(2, 2))
        # post-conv shape = (minibatch_size, 16, 4, 10)
        network = l.layers.Conv2DLayer(
            network,
            num_filters=16,
            filter_size=(3, 3),
            pad=(1, 0),
            W=initial_weights,
            nonlinearity=l.nonlinearities.rectify)
        # post-pool shape = (minibatch_size, 16, 2, 5)
        network = l.layers.MaxPool2DLayer(
            network,
            pool_size=(2, 2))
        # post-conv shape = (minibatch_size, 10, 1, 1)
        network = l.layers.Conv2DLayer(
            network,
            name='bottleneck',
            num_filters=self.bottleneck_width,
            filter_size=(2, 5),
            pad=0,
            W=initial_weights,
            nonlinearity=l.nonlinearities.rectify)
        # post-deconv shape = (minibatch_size, 16, 2, 4)
        network = l.layers.Deconv2DLayer(
            network,
            num_filters=16,
            filter_size=(2, 4),
            stride=(2, 2),
            W=initial_weights)
        # post-deconv shape = (minibatch_size, 16, 4, 11)
        network = l.layers.TransposedConv2DLayer(
            network,
            num_filters=16,
            filter_size=(2, 5),
            stride=(2, 2),
            W=initial_weights)
        # post-deconv shape = (minibatch_size, input_depth, 8, 24)
        network = l.layers.TransposedConv2DLayer(
            network,
            num_filters=self.image_shape[0],
            filter_size=(2, 4),
            stride=(2, 2),
            W=initial_weights)
        return network

    def _setup_cost(self):
        '''Construct the sum-squared loss between the input and the output.
        
        Must be called after self.network is defined.'''
        prediction = l.layers.get_output(self.network)
        cost = l.objectives.squared_error(prediction, self.input_var)
        cost = l.objectives.aggregate(cost, mode='mean')
        return cost

    def _setup_optimizer(self):
        '''Construct the gradient descent optimizer.

        Must be called after self.cost is defined.'''
        weights = l.layers.get_all_params(self.network, trainable=True)
        updates = l.updates.momentum(
            self.cost,
            weights,
            learning_rate=self.learn_rate,
            momentum=0.9)
        return updates
