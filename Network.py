'''This module contains the base class for a generic neural net interface.

Subclassing this class allows us to stay safe against changes in NN frameworks.
'''

class AbstractNetwork(object):
    '''The base neural network class.
    
    This class defines an interface for neural network subclasses.
    '''
    def __init__(self, *args, **kwargs):
        self.network = None
        self.cost = None
        self.optimizer = None

    def fit(self, x_train, y_train):
        raise NotImplemented()

    def predict(self, x):
        raise NotImplemented()

    def extract_layer(self, data, layer):
        raise NotImplemented()

    def preprocess_data(self, x, y):
        raise NotImplemented()
