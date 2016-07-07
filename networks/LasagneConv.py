'''The Lasagne NN module for Daya Bay data'''

import numpy as np
import theano
import theano.tensor as T
import lasagne as l
from operator import mul
import logging
logging.getLogger().setLevel(logging.DEBUG)

from Network import AbstractNetwork
import preprocessing

class IBDPairConvAe(AbstractNetwork):
    '''A convolutional autoencoder for interpreting candidate IBD pairs.'''

    def __init__(self, minibatch_size=128, epochs=1, learn_rate=1e-3,
            bottleneck_width=10, **kwargs):
        '''Initialize a ready-to-train convolutional autoencoder.'''
        super(IBDPairConvAe, self).__init__(self)
        # Shapes are given as (batch, depth, height, width)
        nchannels = kwargs.get('nchannels', 4)
        self.minibatch_shape = (minibatch_size, nchannels, 8, 24)
        self.minibatch_size = minibatch_size
        self.image_shape = self.minibatch_shape[1:-1]
        self.num_features = reduce(mul, self.image_shape)
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.bottleneck_width = bottleneck_width
        self.input_var = T.dtensor4('input')
        self.network = self._setup_network()
        self.train_prediction = self._setup_prediction(deterministic=False)
        self.test_prediction = self._setup_prediction(deterministic=True)
        self.train_cost = self._setup_cost(deterministic=False)
        self.test_cost = self._setup_cost(deterministic=True)
        self.optimizer = self._setup_optimizer()
        self.train_once = theano.function([self.input_var],
            self.train_cost, updates=self.optimizer)
        self.predict_fn = theano.function([self.input_var],
            [self.test_cost, self.test_prediction])

    def _setup_network(self):
        '''Construct the ConvAe architecture for Daya Bay IBDs.'''
        initial_weights = l.init.Normal(1.0/self.num_features, 0)
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
        # post-conv shape = (minibatch_size, bottleneck_width, 1, 1)
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
        network = l.layers.Deconv2DLayer(
            network,
            num_filters=16,
            filter_size=(2, 5),
            stride=(2, 2),
            W=initial_weights)
        # post-deconv shape = (minibatch_size, input_depth, 8, 24)
        network = l.layers.Deconv2DLayer(
            network,
            num_filters=self.image_shape[0],
            filter_size=(2, 4),
            stride=(2, 2),
            W=initial_weights)
        return network

    def _setup_prediction(self, deterministic):
        '''Construct the Theano identifier that refers to the output of the
        autoencoder.'''
        return l.layers.get_output(self.network, deterministic=deterministic)

    def _setup_cost(self, deterministic):
        '''Construct the sum-squared loss between the input and the output.
        
        Must be called after self.network is defined.'''
        if deterministic:
            prediction = self.test_prediction
        else:
            prediction = self.train_prediction
        cost = l.objectives.squared_error(prediction, self.input_var)
        cost = l.objectives.aggregate(cost, mode='mean')
        return cost

    def _setup_optimizer(self):
        '''Construct the gradient descent optimizer.

        Must be called after self.train_cost is defined.'''
        weights = l.layers.get_all_params(self.network, trainable=True)
        updates = l.updates.momentum(
            self.train_cost,
            weights,
            learning_rate=self.learn_rate,
            momentum=0.9)
        return updates


    def minibatch_iterator(self, x, y=None):
        '''Return a generator that goes through the set in mini-batches.
        '''
        if y is not None:
            raise ValueError("We don't need labels here")
            
        def minibatches():
            numinputs = x.shape[0]
            indices = np.arange(numinputs)
            # Shuffle order each time a new set of minibatches is requested
            np.random.shuffle(indices)
            # Check for the small-sample case, in which case we don't use a
            # minibatch
            if numinputs > self.minibatch_size:
                upper = numinputs - self.minibatch_size + 1
            else:
                upper = numinputs
            for i in xrange(0, upper, self.minibatch_size):
                excerpt = slice(i, i + self.minibatch_size)
                yield x[indices[excerpt]]
        return minibatches

    def fit(self, x_train, y_train=None):
        '''Fit and train the autoencoder to the x_train data.'''
        if y_train is not None:
            raise ValueError("We don't need labels here")
        logging.info("Training with %d training samples" % x_train.shape[0])
        for epoch in xrange(self.epochs):
            minibatches = self.minibatch_iterator(x_train)
            for inputs in minibatches():
                cost = self.train_once(inputs)
            logging.info("loss after epoch %d is %f", epoch, cost)

    def predict(self, x, y=None):
        '''Predict the autoencoded image without training.'''
        if y is not None:
            raise ValueError("We don't need labels here")
        return self.predict_fn(x)

    def extract_layer(self, data, layer):
        '''Extract the output of the given layer.'''
        all_layers = l.layers.get_all_layers(self.network)
        # find the layer named with the value of layer
        for one_layer in all_layers:
            if one_layer.name == layer:
                output = l.layers.get_output(one_layer)
                out_fn = theano.function([self.input_var], output)
                return out_fn(data)
        raise ValueError('"%s" is not a layer in our network' % layer)

    def preprocess_data(self, x, y=None):
        '''Prepare the data for the neural network.

            - Remove 0's from the time channels
            - Center the data on 0
            - Scale it to have a standard deviation of 1'''
        std = 1
        preprocessing.fix_time_zeros(x)
        means = preprocessing.center(x)
        stds = preprocessing.scale(x, std, mode='standardize')
        def repeat_transformation(other):
            preprocessing.fix_time_zeros(other)
            other -= means
            other /= stds/std
        return repeat_transformation


class IBDPairConvAe2(IBDPairConvAe):
    '''A CAE based on IBDPairConvAe that scales input and output to be between
       [-1, 1].'''
    def __init__(self, *args, **kwargs):
        '''Initialize an IBDPairConvAe2.'''
        super(IBDPairConvAe2, self).__init__(*args, **kwargs)

    def _setup_network(self):
        '''Set up the IBDPairConvAe network but have it scale output to +/- 1'''
        network = super(IBDPairConvAe2, self)._setup_network()
        # The "network" is really a pointer to the output deconv layer
        network.nonlinearity = l.nonlinearities.tanh
        return network

    def preprocess_data(self, x, y=None):
        '''Prepare the data for the neural network.

            - Remove 0's from the time channels
            - Center the data on 0
            - Scale it to have lie on the interval [-1, 1]'''
        preprocessing.fix_time_zeros(x)
        means = preprocessing.center(x)
        min_, max_, = -1, 1
        mins, maxes = preprocessing.scale_min_max(x, min_, max_)
        def repeat_transformation(other):
            preprocessing.fix_time_zeros(other)
            other -= means
            other -= mins
            other /= maxes - mins
            other *= max_ - min_
            other += min_
        return repeat_transformation

class IBDChargeDenoisingConvAe(IBDPairConvAe2):
    '''A denoising CAE based on IBDPairConvAe2.'''
    def __init__(self, *args, **kwargs):
        '''Initialize the denoising autoencoder.'''
        self.zero_fraction = kwargs.get('zero_fraction', 0.3)
        self.seed = kwargs.get('seed', 498)
        super(IBDChargeDenoisingConvAe, self).__init__(*args, nchannels=2, **kwargs)
        self.only_charge = True

    def _get_corrupt_input(self):
        '''Set up the corrupted input variable with randomly zeroed pixels.'''
        rng = np.random.RandomState(self.seed)
        theano_rng = T.shared_randomstreams.RandomStreams(rng.randint(2 ** 30))
        # each pixel in the mask has a p=1-self.zero_fraction chance of being
        # 1, else 0
        mask = theano_rng.binomial(size=self.input_var.shape,
            dtype=theano.config.floatX, n=1, p=1-self.zero_fraction)
        return mask * self.input_var

    def _setup_network(self):
        '''Set up the IBDPairConvAe2 network, but have it accept only 2 input
        channels, and partly corrupt the input by zeroing out some of the
        pixels.'''
        corrupt_input_layer = l.layers.InputLayer(
            input_var=self._get_corrupt_input(),
            shape=self.minibatch_shape)

        network = super(IBDChargeDenoisingConvAe, self)._setup_network()
        # Reassign the first convolutional layer's input to be the new corrupt
        # input
        layers = l.layers.get_all_layers(network)
        first_conv_layer = layers[1]
        first_conv_layer.input_layer = corrupt_input_layer
        first_conv_layer.input_shape = corrupt_input_layer.output_shape
        return network
