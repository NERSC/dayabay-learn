__author__ = 'racah'

from neon.layers.layer import Layer
import numpy as np

def interpret_in_shape(xshape):
    """
    Helper function to interpret the tensor layout of preceding layer to handle non-recurrent,
    recurrent, and local layers
    """
    if isinstance(xshape, int):
        return (xshape, 1)
    else:
        if len(xshape) == 2:
            return xshape
        else:
            return (np.prod(xshape), 1)

class DataTransform(Layer):

    """
    A layer that applies a specified transform to input data in fprop only.
    Only supported as the first layer in the network.
    Arguments:
        transform (Transform): a transform object with fprop function to apply
        name (str, optional): Layer name. Defaults to "DataTransformLayer"
    """

    def __init__(self, transform, name="DataTransformLayer"):
        super(DataTransform, self).__init__(name)
        self.transform = transform
        self.owns_output = False

    def __str__(self):
        return "DataTransform Layer '%s': %s" % (
               self.name, self.transform.__class__.__name__)

    def configure(self, in_obj):
        super(DataTransform, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nout, _) = interpret_in_shape(self.in_shape)
        return self

    def fprop(self, inputs, inference=False):
        self.outputs = self.inputs = inputs
        self.outputs[:] = self.transform(self.inputs)
        return self.outputs

    def bprop(self, *args):
        return None