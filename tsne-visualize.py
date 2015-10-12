
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import h5py
import subprocess
import os
from tsne_source_code import tsne

base_path='/scratch3/scratchdirs/jialin/dayabay/'
#base_path='./'
points_per_class=200

file_number = 29155
event_types = ['muo', 'fla', 'oth']
post_process_types = ['ae', 'raw']

ae = {'path': base_path + 'autoencoded', 'prefix':'output-', 'suffix': '', 'h5_key': 'autoencoded'}
raw = {'path': base_path + 'preprocess', 'prefix':'output-', 'suffix': '', 'h5_key': 'inputs'}
post_proc_info = {'ae': ae, 'raw': raw}


for pp_key in post_process_types:
    pp_dict = post_proc_info[pp_key]
    for event in event_types:
        file_path = os.path.join(pp_dict['path'], pp_dict['prefix'] + event  + pp_dict['suffix'], str(file_number) + '.h5')
        pp_dict[event] = h5py.File(file_path,'r')[pp_dict['h5_key']][:points_per_class]
        pp_dict[event + '_key'] = pp_dict[event][0, -1]



    pp_dict['combined'] = np.vstack(tuple(pp_dict[k] for k in event_types))

    pp_dict['tsne'] = tsne.tsne(pp_dict['combined'][:, :-1].astype('float64'), 2, pp_dict['combined'].shape[0], 30.0) #, 2, 10, 10.0) #b/c the last column is the label

    ts = pp_dict['tsne']
    comb = pp_dict['combined']

    colors = ['b','r','g', 'k', 'm']
    markers = ['o','o','o','s', 's']
    plt.figure(1)
    plt.clf()
    for i, event in enumerate(event_types):
        ev = pp_dict[event + '_key']
        plt.scatter(ts[comb[:, -1] == ev][:, 0], ts[comb[:, -1] == ev][:, 1], marker=markers[i], c=colors[i], alpha=0.9, label=event)
    plt.legend(loc='upper right')
    plt.savefig('./tsne-%s.pdf' % (pp_key))

