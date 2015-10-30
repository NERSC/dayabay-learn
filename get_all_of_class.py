__author__ = 'racah'
#gets all data of a particular class from peter's initially specified single files
#puts it into intel fomrat which is data | class label (as 1-5 number) in last column
#useful for old version of neon b/c points get shuffled around from output

import h5py
import numpy as np
import sys
import os

hash_table = {'adinit' : 1, 'addelay' : 2, 'muo' : 3, 'fla' : 4, 'oth' : 5}
rev_hash_table = {hash_table[k]: k for k in hash_table.keys()}
des_class, h5_path, save_dest = sys.argv[1:]
h5_file_name = os.path.splitext(os.path.basename(h5_path))[0] + '.h5'

if des_class.isdigit():
    des_cl = int(des_class)
else:
    des_cl = hash_table[des_class]



h5f = h5py.File(h5_path, 'r')
data = h5f['inputs']
labels = h5f['targets']

#class 1 has a 1 at 0th element in one-hot encoding, class 2 has a 1 at 1th elemtent, etc.
des_data = data[:][labels[:, des_cl - 1] == 1.0]
des_data = des_data[np.sum(des_data, axis=1) != 0.]
rows = des_data.shape[0]
final_data = np.hstack((des_data, des_cl * np.ones((rows,1))))
save_name = os.path.join(save_dest, h5_file_name)

h5f_o = h5py.File(save_name, 'w')
h5f_o.create_dataset('inputs', data=final_data)
h5f_o.close()

