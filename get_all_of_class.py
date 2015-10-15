__author__ = 'racah'
#gets all data of a particular class from peter's initially specified single files
#puts it into intel fomrat which is data | class label (as 1-5 number) in last column
#useful for old version of neon b/c points get shuffled around from output

import h5py
import numpy as np
import sys
import os

des_cl, h5_path, save_dest = sys.argv[1:]
des_cl = int(des_cl)

h5_file_name = os.path.basename(h5_path)

h5f = h5py.File(h5_path, 'r')
data = h5f['inputs']
labels = h5f['targets']

#class 1 has a 1 at 0th element in one-hot encoding, class 2 has a 1 at 1th elemtent, etc.
des_data = data[:][labels[:, des_cl - 1] == 1.0]

save_name = os.path.join(save_dest, h5_file_name)

h5f_o = h5py.File(save_name, 'w')
h5f_o.create_dataset('inputs', data=des_data)
h5f_o.close()

