__author__ = 'racah'
import h5py
import numpy as np
import os
import sys

def duplicates(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff == 0).all(axis=1)
    return a[ui]

def get_zeros(a):
     return np.where(~a.any(axis=1))[0]

if __name__ == "__main__":

    base_path='./'
    points_per_class=200

    file_number = sys.argv[1]
    event_types = ['muo', 'fla', 'oth']#, 'adinit', 'addelay']
    post_process_types = ['ae', 'raw']

    ae = {'path': base_path + 'autoencoded', 'prefix':'output-', 'suffix': '', 'h5_key': 'autoencoded'}
    raw = {'path': base_path + 'preprocess', 'prefix':'output-', 'suffix': '', 'h5_key': 'inputs'}
    post_proc_info = {'ae': ae, 'raw': raw}

    for pp_key in post_process_types:
        pp_dict = post_proc_info[pp_key]
        for event in event_types:
            file_path = os.path.join(pp_dict['path'], pp_dict['prefix'] + event  + pp_dict['suffix'], str(file_number) + '.h5')
            data = h5py.File(file_path,'r')[pp_dict['h5_key']][:,:-1]
            print "For %s-%s: %s there are %i zero rows and %i duplicate rows out of %i total rows"%(pp_key,event, file_path,get_zeros(data).shape[0], duplicates(data).shape[0]-1,data.shape[0])




