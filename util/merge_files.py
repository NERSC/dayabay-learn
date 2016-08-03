


import h5py

import os
path_prefix="/global/homes/s/skohn/ml/dayabay-data-conversion/extract_ibd"
h5files = [path for path in os.listdir(path_prefix) if "ibd_yasu" in path]

def comp_fnames(p1,p2):
    n1,n2 = map(lambda x: int(x.split('_')[2]),[p1,p2])
    if n1 > n2:
        ret = 1
    elif n2 > n1:
        ret = -1
    else:
        ret = 0
    return ret
        
    
    
h5files.sort(cmp=comp_fnames)

h5paths = [os.path.join(path_prefix, h5file) for h5file in h5files]


for h5path in h5paths:
    assert os.path.exists(h5path)

master_file = h5py.File('/project/projectdirs/dasrepo/ibd_pairs/all_pairs.h5')

sample_h5file = h5py.File(h5paths[0])
key = sample_h5file.keys()[0]
cols = sample_h5file[key].shape[1]
rows = sum([h5py.File(h5path)[key].shape[0] for h5path in h5paths])  
dtype = sample_h5file[key].dtype

print dtype

master_dataset = master_file.create_dataset(name=key, shape=(rows,cols), dtype=dtype)



for i, h5path in enumerate(h5paths):
    #print h5path
    #print i *10000, (i+1)*10000
    h5d = h5py.File(h5path)[key]
    master_dataset[i *10000 :(i+1)*10000] = h5d

master_dataset

import numpy as np
#master_dataset[]

#inds = np.random.randint(0,1000000,1000)
low = 0
high  =10000


np.all(np.equal(master_dataset[low:high], h5py.File(h5paths[0])[key][low:high]))



master_file.close()

