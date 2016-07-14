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
sys.path.append(os.path.abspath('../networks'))
from data_loaders import load_ibd_pairs, load_predictions
from LasagneConv import IBDPairConvAe, IBDPairConvAe2
from LasagneConv import IBDChargeDenoisingConvAe

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default=None, help='file to read')
    parser.add_argument('-e', '--event', default=0, type=int,
        help='event index to read')
    gp = parser.add_mutually_exclusive_group(required=True)
    gp.add_argument('-i', '--input', action='store_true',
        help='interpret the file as input')
    gp.add_argument('-o', '--output', action='store_true',
        help='interpret the file as output')
    parser.add_argument('--flex-color', action='store_true',
        help='use a variable color scale per plot')
    parser.add_argument('--preprocess', default=None, choices=[
            None,
            'IBDPairConvAe',
            'IBDPairConvAe2',
            'IBDChargeDenoisingConvAe',
        ],
        help='preprocess with the given network')
    parser.add_argument('--save', default=None,
        help='Save the plot(s) to a file')
    parser.add_argument('-n', '--num-events', default=1, type=int,
        help='If saving, how many events to save')
    args = parser.parse_args()

    include_time = {
        'IBDPairConvAe': True,
        'IBDPairConvAe2': True,
        'IBDChargeDenoisingConvAe': False,
    }

    # Figure out a good number of pairs to retrieve
    # Minimum of 200, then ensure there are enough to get to all of the desired
    # events.
    num_pairs = args.event + args.num_events + 200
    if args.file is None:
        infile = ('/project/projectdirs/dasrepo/ibd_pairs/all_pairs.h5')
    else:
        infile = args.file
    if args.input:
        data, _, _ = load_ibd_pairs(infile, train_frac=1, valid_frac = 0,
            tot_num_pairs=num_pairs)
        if args.preprocess is not None:
            conv_class = eval(args.preprocess)
            cae = conv_class()
            cae.preprocess_data(data)
    elif args.output:
        data = load_predictions(infile, tot_num_pairs=num_pairs)

    if args.save is None:
        args.num_events = 1

    for i in range(args.event, args.event + args.num_events):
        event = data[i]
        # Construct pyplot canvas with images
        image_args = {
            'interpolation': 'nearest',
            'aspect': 'auto',
            'cmap': plt.get_cmap('PuBu')
        }
        if not args.flex_color:
            image_args.update({
                'vmin': -1,
                'vmax': 1,
            })
        fig = plt.figure(1)
        if include_time[args.preprocess]:
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
        else:
            prompt_charge_ax = plt.subplot(2, 1, 1)
            prompt_charge_im = plt.imshow(event[0], **image_args)
            prompt_charge_ax.set_title('Prompt Charge')
            plt.colorbar()
            delayed_charge_ax = plt.subplot(2, 1, 2)
            delayed_charge_im = plt.imshow(event[2], **image_args)
            delayed_charge_ax.set_title('Delayed Charge')
            plt.colorbar()

        if args.save is None:
            plt.show()
        else:
            split = os.path.splitext(args.save)
            plt.savefig("%s_%d%s" % (split[0], i, split[1]))
            plt.clf()
