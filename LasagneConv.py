'''The Lasagne NN module for Daya Bay data'''

import numpy as np
import theano
import theano.tensor as T
import lasagne as l

import Network.AbstractNetwork as AbstractNetwork

class IBDPairConvAe(AbstractNetwork):
    '''A convolutional autoencoder for interpreting candidate IBD pairs.'''

    def __init__(self):
        '''Initialize a ready-to-train convolutional autoencoder.'''
        super(IBDPairConvAe, self).__init__(self)
        self.image_shape = (4, 8, 24)
        self.minibatch_size = 128
        self.learn_rate = 0.0001
        self.bottleneck_width = 10
        self.input_var = T.dtensor4('input')
        self.network = self._setup_network()
        self.cost = self._setup_cost()
        self.optimizer = self._setup_optimizer()

    def _setup_network(self):
        '''Construct the ConvAe architecture for Daya Bay IBDs.'''
        initial_weights = l.init.Normal(1, 0)
        network = l.layers.InputLayer(
            shape=(self.minibatch_size, *self.image_shape)
            input_var=self.input_var)
        network = l.layers.Conv2DLayer(
            network,
            num_filters=16,
            filter_size=(5, 5),
            pad=(2, 2),
            W=initial_weights,
            nonlinearity=l.nonlinearities.rectify)
        network = l.layers.MaxPool2DLayer(
            network,
            pool_size=(2, 2))
        network = l.layers.Conv2DLayer(
            network,
            num_filters=16,
            filter_size=(3, 3),
            pad=(0, 1),
            W=initial_weights,
            nonlinearity=l.nonlinearities.rectify)
        network = l.layers.MaxPool2DLayer(
            network,
            pool_size=(2, 2))
        network = l.layers.Conv2DLayer(
            name='bottleneck',
            num_filters=self.bottleneck_width,
            filter_size=(2, 5),
            pad=0,
            W=initial_weights,
            nonlinearity=l.nonlinearities.rectify)
        network = l.layers.Deconv2DLayer(
            network,
            num_filters=16,
            filter_size=(2, 4),
            stride=(2, 2),
            W=initial_weights)
        network = l.layers.Deconv2DLayer(
            network,
            num_filters=16,
            filter_size=(2, 5),
            stride=(2, 2),
            W=initial_weights)
        network = l.layers.Deconv2DLayer(
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
        cost = aggregate(loss, mode='mean')
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
