import numpy as np
import os


def center(data):
    '''Subtract the mean over all samples and pixels, independent by channel.

    Expects data of the form [batch, channel, height, width].'''
    means = data.mean(axis=(0, 2, 3), keepdims=True)
    data -= means
    return means.flatten()

def scale(data, std=None, mode="standardize"):
    '''Scale the data to the given std over all samples and pixels, independent
    by channel.

    Expects data of the form [batch, channel, height, width].'''
    
    if mode == "standardize":
        stds = data.std(axis=(0, 2, 3), keepdims=True)
        data /= stds/std
        return stds.flatten()
    elif mode == "normalize":
        scale_min_max(data)
    else:
        raise NotImplementedError

def channelNeighbors(points, full, axes=(2, 3), condition=lambda x:True,
        stillbad = None):
    '''Create a list of orthogonal neighbors of each point that satisfy the given
    condition.

    return_val[i] = [(neighbor 1 index), (neighbor 2 index), ...] for points[i]

    Axes specifies the directions to explore for neighbors. For example,
    specifying axes=(2, 3) will hold the first 2 indices constant and only
    adjust the 2nd and 3rd indices.

    Condition is a function of the potential neighbor. For example it can test
    if the neighbor is nonzero. Any neighbor for which the condition returns
    True is included.
    
    Stillbad is an optional list to hold point indices that have no neighbors
    satisfying condition. If provided, it should be specified as an empty list.'''
    # adjust the condition to apply to the data at an index rather than at the
    # index itself (as filter() would normally assume)
    condition_full = lambda x:condition(full[x])
    shape = full.shape
    all_neighbors = []
    for point in points:
        point_neighbors = []
        for axis in axes:
            test_point_low = np.asarray(point)
            test_point_up = test_point_low.copy()
            if test_point_low[axis] > 0:
                test_point_low[axis] -= 1
                point_neighbors.append(tuple(test_point_low))
            if test_point_up[axis] < shape[axis] - 1:
                test_point_up[axis] += 1
                point_neighbors.append(tuple(test_point_up))
        good_neighbors = filter(condition_full, point_neighbors)
        if len(good_neighbors) == 0:
            if stillbad is not None:
                stillbad.append(point)
        all_neighbors.append(good_neighbors)
    return all_neighbors

def fix_time_zeros(data):
    bads = zip(*np.nonzero(data == 0))
    # Only take the time channels (not the charge channels)
    bads = filter(lambda x:x[1] in (1, 3), bads)


    # Get list of neighbors for each point
    # [[point 0 neighbors], [point 1 neighbors], ...]
    stillbad = []
    neighbors = channelNeighbors(bads, data, axes=(2, 3),
        condition=lambda x:x < 0, stillbad=stillbad)
    replacements = np.hstack(np.mean(data[zip(*ns)]) if len(ns) > 0 else 0 for ns in neighbors)
    data[zip(*bads)] = replacements
    iterations = 0
    while len(stillbad) > 0 and iterations < 3:
        iterations += 1
        bads = stillbad
        stillbad = []
        neighbors = channelNeighbors(bads, data, axes=(2, 3),
            condition=lambda x:x < 0, stillbad=stillbad)
        replacements = np.hstack(np.mean(data[zip(*ns)]) if len(ns) > 0 else 0 for ns in neighbors)
        data[zip(*bads)] = replacements
    
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

