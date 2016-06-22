'''This script will plot the PMT hit maps of the specified IBD pair'''
import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse
import sys
import os
import logging
logging.getLogger().setLevel(logging.DEBUG)
sys.path.append(os.path.abspath('../util'))
from data_loaders import load_ibd_pairs
from helper_fxns import center, scale

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

def smearTimeZeros(data):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default=None, help='file to read')
    parser.add_argument('-i', '--index', default=0, type=int,
        help='event index to read')
    parser.add_argument('--no-preprocess', action='store_true',
        help='turn off data centering and scaling')
    parser.add_argument('--no-fix-zeros', action='store_true',
        help='turn of zero replacement')
    args = parser.parse_args()

    if args.file is None:
        infile = ('/project/projectdirs/dasrepo/ibd_pairs/all_pairs.h5')
    else:
        infile = args.file
    data, _, _ = load_ibd_pairs(infile, train_frac=1, valid_frac = 0,
    tot_num_pairs=1000)

    if not args.no_fix_zeros:
        # Before adjusting values, get rid of zeros in time channels by replacing
        # them with the average of the neighboring cells
        smearTimeZeros(data[:1000])


    if not args.no_preprocess:
        center(data[:1000])
        scale(data[:1000], 1)

    event = data[args.index]
    # Construct pyplot canvas with images
    image_args = {
        'interpolation': 'nearest',
        'aspect': 'auto',
        'cmap': plt.get_cmap('spectral')
    }
    fig = plt.figure(1)
    prompt_charge_ax = plt.subplot(2, 2, 1)
    prompt_charge_im = plt.imshow(event[0], **image_args)
    prompt_charge_ax.set_title('Prompt Charge')
    plt.colorbar()
    prompt_time_ax = plt.subplot(2, 2, 2, sharey=prompt_charge_ax)
    prompt_time_im = plt.imshow(event[1], **image_args)
    prompt_time_ax.set_title('Prompt Time')
    plt.colorbar()
    delayed_charge_ax = plt.subplot(2, 2, 3, sharex=prompt_charge_ax)
    delayed_charge_im = plt.imshow(event[2], **image_args)
    delayed_charge_ax.set_title('Delayed Charge')
    plt.colorbar()
    delayed_time_ax = plt.subplot(2, 2, 4, sharex=prompt_time_ax,
        sharey=delayed_charge_ax)
    delayed_time_im = plt.imshow(event[3], **image_args)
    delayed_time_ax.set_title('Delayed Time')
    plt.colorbar()

    plt.show()
