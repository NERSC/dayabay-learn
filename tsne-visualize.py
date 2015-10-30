
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import h5py
import subprocess
import os
from tsne_source_code import tsne
import pickle
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
#base_path='/scratch3/scratchdirs/jialin/dayabay/'

final_dim = 2
perp = 50.0
max_iter = 1000
base_path='./'
poinx_ts_per_class=200

file_name = 'single_20000'
event_types = ['muo', 'fla', 'oth', 'adinit', 'addelay']
post_process_types = ['ae', 'raw']

ae = {'path': base_path + 'autoencoded', 'prefix':'output-', 'suffix': '', 'h5_key': 'autoencoded'}
raw = {'path': base_path + 'preprocess', 'prefix':'output-', 'suffix': '', 'h5_key': 'inputs'}
post_proc_info = {'ae': ae, 'raw': raw}


for pp_key in post_process_types:
    print pp_key
    pp_dict = post_proc_info[pp_key]
    for event in event_types:
        file_path = os.path.join(pp_dict['path'], pp_dict['prefix'] + event  + pp_dict['suffix'], file_name + '.h5')
        pp_dict[event] = h5py.File(file_path,'r')[pp_dict['h5_key']][:poinx_ts_per_class]
        pp_dict[event + '_key'] = pp_dict[event][0, -1]



    x_labeled = np.vstack(tuple(pp_dict[k] for k in event_types))

    print x_labeled.shape

    #remove labels from x_labeled
    X = x_labeled[:, :-1]
    pkled_data_name = "%s-%s-%s-%i-%f-%i"%(pp_key,'-'.join(event_types), file_name, final_dim, perp, max_iter)
    full_path = os.path.join('pkled_tsne', pkled_data_name)
    if os.path.exists(full_path):
        print "reconstructing tsne from pickled file. This shouldn't take long..."
        x_ts = pickle.load(open(full_path))
    else:
        x_ts = tsne.tsne(X[:,:-1].astype('float64'), final_dim, X.shape[1], perp, max_iter=max_iter) #, 2, 10, 10.0
        pickle.dump(x_ts, open(full_path, 'w'))

    pca = PCA(n_components=final_dim)
    x_pca = tsne.pca(X, final_dim)#pca.fit_transform(X)


    colors = ['b', 'r', 'g', 'y', 'm']
    markers = ['o', 'o', 'o', 's', 's']
    kwargs={}
    if final_dim == 3:
        kwargs['projection']='3d'
    reduced_arrs = {'tsne': x_ts} #, 'pca':x_pca}
    for red_type, x_red in reduced_arrs.iteritems():
        fig = plt.figure(1)
        plt.clf()
        ax = fig.add_subplot(111, **kwargs)
        for i, event in enumerate(event_types):
            # if event == 'muo':
            #     continue
            ev_num = pp_dict[event + '_key']
            print x_red.shape
            x_red_ev = x_red[x_labeled[:, -1] == ev_num]
            print x_red_ev.shape
            if final_dim == 3:
                ax.scatter(x_red_ev[:, 0], x_red_ev[:, 1], x_red_ev[:, 2], marker=markers[i], c=colors[i], alpha=0.9, label=event)
            else:
                ax.scatter(x_red_ev[:, 0], x_red_ev[:, 1], marker=markers[i], c=colors[i], alpha=0.9, label=event)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])


        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, shadow=True, ncol=5, prop={'size': 6})
        plt.savefig('./plots/%s-%s-%iD.pdf' % (red_type, pp_key, final_dim))

