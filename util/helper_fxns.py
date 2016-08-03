


import numpy as np
import os
import sys
from sklearn.manifold import TSNE
import pickle
def create_run_dir():
    results_dir = './results/ev-runs'
    run_num_file = os.path.join(results_dir, "run_num.txt")
    if not os.path.exists(results_dir):
        print "making results dir"
        os.mkdir(results_dir)

    if not os.path.exists(run_num_file):
        print "making run num file...."
        f = open(run_num_file,'w')
        f.write('0')
        f.close()




    f = open(run_num_file,'r+')

    run_num = int(f.readline()) + 1

    f.seek(0)

    f.write(str(run_num))


    run_dir = os.path.join(results_dir,'run%i'%(run_num))
    os.mkdir(run_dir)
    return run_dir



def make_accidentals(only_charge=True, fraction=0.5, *datasets):
    '''Scramble a given fraction of events in datasets to make them
    "accidental" background.

    Accomplish this task by shuffling prompt signals (charge and possibly time,
    depending on the value of only_charge) to produce uncorrelated hit
    patterns.

    This method assumes the following shape for supplied data: (batch, [prompt
    charge, prompt time, delayed charge, delayed time], x, y).'''
    if fraction == 0:
        return
    for data in datasets:
        totalentries = data.shape[0]
        num_scrambled = int(np.ceil(totalentries * fraction))
        toscramble = np.random.permutation(totalentries)[:num_scrambled]
        scrambledestinations = np.random.permutation(toscramble)
        data[scrambledestinations, 0] = data[toscramble, 0]
        if not only_charge:  # then also scramble time
            data[scrambledestinations, 1] = data[toscramble, 1]
        return









