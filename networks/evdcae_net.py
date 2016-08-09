


import sys
import theano
import os
import time
import numpy as np
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import lasagne
from lasagne import layers as L
import matplotlib
import matplotlib.pyplot as plt
import h5py

import time
import preprocessing
from sklearn.manifold import TSNE

#temporary!
if __name__ == "__main__":
    #just a little trick for testing in order to get the data loader
    sys.path.insert(0,os.path.dirname(os.getcwd()))
    

from util.data_loaders import get_ibd_data
from networks.Network import AbstractNetwork
#enable importing of notebooks
from train_val import train
from networks.preprocessing import scale, scale_min_max



class DenoisingConvAe(AbstractNetwork):
    def __init__(self, network_kwargs, save_dir='./results', load_path=None):
        self.network_kwargs = network_kwargs
        self.save_dir = save_dir
        
        self.train_fn,         self.val_fn,         self.pred_fn,         self.hlayer_fn,        self.salmap_fn,        self.network = build_network(**self.network_kwargs)
        
        

    def fit(self, x_train, y_train, x_val,y_val):
        
        x_train, x_val = self.preprocess_data(x_train, x_val)
        train((x_train, x_train, x_val,x_val),
                             self.network, 
                             self.train_fn, 
                             self.val_fn,
                             hlayer_fn = self.hlayer_fn,
                             pred_fn = self.pred_fn,
                             salmap_fn = self.salmap_fn,
                             epochs=self.network_kwargs['epochs'],
                             batchsize=self.network_kwargs['batch_size'],
                             save_path = self.save_dir)
        

    def predict(self, x):
        return self.pred_fn(x)
        

    def extract_hidden_layer(self, data):
        return self.hlayer_fn(data)
    
    def get_saliency_map(self,data):
        return self.salmap_fn(data)
        

    def minibatch_iterator(self, x, y):
        '''iterate over minibatches'''
        raise NotImplemented()

    def preprocess_data(self, train,val ):
        mins, maxes = scale_min_max(train)
        scale_min_max(val,mins=mins,maxes=maxes)
        return train, val



def build_network(learning_rate = 0.01,
                  input_shape=(None,2,8,24),
                  momentum = 0.9,
                  num_filters=128,
                  num_fc_units=1024,
                  num_extra_conv=0, 
                  num_pool=4,
                  nonlinearity=lasagne.nonlinearities.rectify,
                  w_init=lasagne.init.HeNormal(),
                  dropout_p=0.,
                  corruption_p = 0.3,
                  load=False,
                 **unused_kwargs):
    
    input_var = T.tensor4('input_var')
    target_var = T.tensor4('target_var')
    print("Building model and compiling functions...")
    
    
    network, hid_layer = build_denoising_convae(input_var,
                                                input_shape,
                                                num_filters,
                                                num_fc_units,
                                                num_extra_conv, 
                                                num_pool,
                                                nonlinearity,
                                                w_init,
                                                dropout_p,
                                                corruption_p)
    
    if load:
        with np.load('model.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(network, param_values)


    prediction = lasagne.layers.get_output(network, deterministic=False)
    hid_layer_output = lasagne.layers.get_output(hid_layer, deterministic=True)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()
    
  

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=momentum)


    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction,
                                                                target_var)
    test_loss = test_loss.mean()


    salmap = theano.grad(hid_layer_output.sum(), wrt=input_var)

    test_acc = test_loss 


    train_fn = theano.function([input_var, target_var], loss, updates=updates)



    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    
    pred_fn = theano.function([input_var], test_prediction)
    
    hlayer_fn = theano.function([input_var], hid_layer_output )
    
    salmap_fn = theano.function([input_var], salmap)

    return train_fn, val_fn, pred_fn, hlayer_fn, salmap_fn, network

def build_denoising_convae(input_var,input_shape,
                                  num_filters,
                                  num_fc_units,
                                  num_extra_conv, 
                                  num_pool,
                                  nonlinearity,
                                  w_init,
                                  dropout_p,
                                  corruption_p):
   
    

    
    rng = np.random.RandomState(498)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    #do denoising here
    corrup_input = theano_rng.binomial(size=input_var.shape, n=1,
                                        p=1 - corruption_p,
                                        dtype=theano.config.floatX) * input_var
    
    
    
    #input var is (n_ex x 2 x 8 x 24)
    network = L.InputLayer(shape=input_shape, input_var=corrup_input)
    print network.get_output_shape_for(input_shape)
    #output of this is num_filters x 11 x 12
    network = L.Conv2DLayer(network, 
                            num_filters=num_filters, 
                            filter_size=(2,2),
                            pad=(2,0),
                            stride=(1,2),
                            nonlinearity=nonlinearity,
                            W=w_init)
    print network.get_output_shape_for(network.input_shape)
    
    #output of this is num_filters x 6 x 6
    network = L.Conv2DLayer(network, num_filters=num_filters, 
                                 filter_size=(2,2),
                                 pad=(1,0),
                                 stride=2)
    
    last_conv_shape = network.get_output_shape_for(network.input_shape)
    print last_conv_shape
    
    #output of this is num_fc_units
    network = lasagne.layers.DenseLayer(
                                lasagne.layers.dropout(network, p=dropout_p),
                                num_units=num_fc_units,
                                nonlinearity=nonlinearity)
    print network.get_output_shape_for(network.input_shape)
    #capture hidden layer
    hid_layer = network
    
    #output of this is num_filters*6*6
    network = lasagne.layers.DenseLayer(
                                lasagne.layers.dropout(network, p=dropout_p),
                                num_units=np.prod(last_conv_shape[1:]),
                                nonlinearity=nonlinearity)
    print network.get_output_shape_for(network.input_shape)
    
    #output of this is num_filters x 6 x 6
    network = lasagne.layers.ReshapeLayer(network, shape=([0],last_conv_shape[1],last_conv_shape[2],last_conv_shape[3]))
    
    print network.get_output_shape_for(network.input_shape)
    #output of this is num_filters x 11 x 12
    network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_filters, 
                                 filter_size=(3,2),
                                 crop=(1,0),
                                 stride=(2,2),
                                 nonlinearity=nonlinearity,
                                 W=w_init)
    
    print network.get_output_shape_for(network.input_shape)
    #output of this is num_filters x 8  x 24
    #note the number of filters has to be same as number of input channels
    network = lasagne.layers.TransposedConv2DLayer(network, num_filters=input_shape[1], 
                                 filter_size=(2,2),
                                 crop=(2,0),
                                 stride=(1,2),
                                 nonlinearity=lasagne.nonlinearities.linear,
                                 W=w_init)
    print network.get_output_shape_for(network.input_shape)

    
    return network, hid_layer



if __name__ == "__main__":
    #make data
    xtr,xv,xte = get_ibd_data(tot_num_pairs=30, preprocess=True)
    train_fn, val_fn, pred_fn, hlayer_fn, salmap_fn, network = build_network(input_shape=(None,4,8,24))
    a=train_fn(xtr,xtr)
    b=pred_fn(xtr)
    c=hlayer_fn(xtr)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    plt.imshow(b[0,0],interpolation='none')
    plt.colorbar()
    plt.figure(2)
    plt.imshow(xtr[0,0],interpolation='none')
    plt.colorbar()

if __name__ == "__main__":
    xtr,xv,xte = get_ibd_data(tot_num_pairs=30, preprocess=True)
    for x in [xtr,xv,xte]:
        assert(x.min() > -1.1)
        assert (x.max() < 1.1)





if __name__ == "__main__":
    x_train, x_val, x_test = get_ibd_data(tot_num_pairs=200, preprocess=True, just_charges=True)

    dca = DenoisingConvAe(network_kwargs={'learning_rate':0.01}, 
                          train_kwargs={'num_epochs': 11, 'save_path': os.path.dirname(os.getcwd()) + '/results'})

    dca.fit(x_train,x_train,x_val,x_val)

    rec= dca.predict(x_train)

    hlayer = dca.extract_hidden_layer(x_train)

#     salmap =

    ts = TSNE(perplexity=50).fit_transform(hlayer)

    plt.scatter(ts[:,0], ts[:,1])

