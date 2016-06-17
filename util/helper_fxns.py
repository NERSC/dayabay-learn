import numpy as np
import os


#getting rid of the in place option (just complicates things. When would we not do this in place?)
def center(data):
    '''Subtract the mean over all samples and pixels, independent by channel.

    If in_place, update the given data array and return the array of means.
    Else, return (array of means, new array).

    Expects data of the form [batch, channel, height, width].'''
    means = data.mean(axis=(0, 2, 3), keepdims=True)
    #due to numpy objects passed by reference (almost forgot about this)
    data -= means
    return means.flatten()

def scale(data, std=None, mode="standardize"):
    '''Scale the data to the given std over all samples and pixels, independent
    by channel.

    If in_place, update the given data array and return the array of stds.
    Else, return (array of stds, new array).

    Expects data of the form [batch, channel, height, width].'''
    
    if mode == "standardize":
        stds = data.std(axis=(0, 2, 3), keepdims=True)
        data /= stds/std
        return stds.flatten()
    elif mode == "normalize":
        scale_min_max(data)
    else:
        raise NotImplementedError
    
def scale_min_max(data, min_=-1, max_=1):
    '''scales data to be between min and max in place'''
    mins = data.min(axis=(0, 2, 3), keepdims=True)
    maxes = data.max(axis=(0, 2, 3), keepdims=True)

    #data = 2 * ((data - mins) / (maxes - mins)) - 1
    #in place
    data -= mins
    data /= (maxes-mins)
    data *= 2
    data -= 1
    
def filter_out_zeros(X,y):
    
    #get indices of all rows that don't contain all zeroes
    nonzero_rows = ~np.all(X[:, : ]==0., axis=1)
    
    #filter for these nonzero rows
    X = X[nonzero_rows]
    y = y[nonzero_rows]
    return X,y

def get_equal_per_class(X,y, nclass):
#     #get the number of rows in X that correspond to a y for each given class, then take the class that 
#     #corresponds to the fewest rows in X
#     min_classes = [X[y == float(cl)].shape[0] for cl in range(nclass)]
#     print min_classes
#     min_class_count = min(min_classes)
#     #get min_class_count rows of X for each corresponding class
#     print y.shape
#     X = np.vstack(tuple([X[y == float(cl)][:min_class_count, :] for cl in range(nclass)]))
#     y = np.vstack(tuple([y[y == float(cl)][:min_class_count] for cl in range(nclass)]))
#     print y.shape
    
    return X,y






