
import numpy as np
import numpy as np
import h5py
import subprocess
import os
from tsne_python import tsne
from bh_tsne import bhtsne
from matplotlib import pyplot as plt

base_path='/scratch3/scratchdirs/jialin/dayabay/'
points_per_class=100

file_number = 29155
event_types = ['muo', 'fla', 'oth']
post_process_types = ['ae', 'raw']

ae = {'path': base_path + 'autoencoded', 'prefix':'output-', 'suffix': '', 'h5_key': 'autoencoded'}
raw = {'path': base_path, 'prefix':'output-', 'suffix': '', 'h5_key': 'inputs'}
post_proc_info = {'ae': ae, 'raw': raw}


for pp_key, pp_dict in post_proc_info .iteritems():
    for event in event_types:
        file_path = os.path.join(pp_dict['path'], pp_dict['prefix'] + event  + pp_dict['suffix'], str(file_number) + '.h5')
        pp_dict[event] = h5py.File(file_path,'r')[ae['h5_key']][:points_per_class]
        pp_dict[event + '_key'] = pp_dict[event][0, -1]



    pp_dict['combined'] = np.vstack(tuple(pp_dict[k] for k in event_types))

    pp_dict['tsne'] = tsne.tsne(ae['combined'][:, :-1]) #, 2, 10, 10.0) #b/c the last column is the label


    colors = ['b','r','g', 'k', 'm']
    markers = ['o','o','o','s', 's']
    plt.figure(1)
    for i, event in enumerate(event_types):
        tsne_event = pp_dict['tsne'][ pp_dict['combined'][:, -1] == pp_dict[event + '_key']]
        plt.scatter(tsne_event[:, 0], tsne_event[:, 1], marker=markers[i], c=colors[i], alpha=0.5, label=event)
    plt.legend(loc='upper right')
    plt.savefig('./tsne-%s.jpg' % (pp_key))

