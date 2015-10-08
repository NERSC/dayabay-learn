import sys
import h5py
import numpy as np
import pickle
import os

save_dir = sys.argv[1]
pkl = sys.argv[2]
final = np.zeros((0, 11))
ae = pickle.load(open(pkl))
#labels should come in order
label = float(os.path.basename(pkl).split('.')[0].split('-')[1])

#make column of labels
labels = label * np.ones((ae.shape[0],1))

#add labelled column
ae = np.hstack((ae,labels ))

#stack with previous
final = np.vstack((final, ae))


#puts output in same directory as first pkl input
output_filename = os.path.basename(pkl).split('.')[0].split('-')[0] + '.h5'
h5f = h5py.File(os.path.join(save_dir, output_filename), 'w')
h5f.create_dataset('autoencoded', data=final)
h5f.close()



