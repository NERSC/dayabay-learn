
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
import sys
from matplotlib import cm
#base_path='/scratch3/scratchdirs/jialin/dayabay/'


# 1) Primary AD           10000
# 2) Delayed AD response  01000
# 3) Muon decay           00100
# 4) Flasher              00010
# 5) Other (background noise) 00001

from matplotlib import pyplot as plt
raw_dim=192
def save_image(vec, name, y_offset=0.3, x_offset=0.69, font_size=5):
    plt.clf()
    im = vec.reshape(8, 24)
    ax=plt.imshow(im, interpolation='none')
    left, right, bottom, top = ax.get_extent()
    plt.colorbar(orientation="horizontal")
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            plt.text(left + y_offset + y, top + x_offset + x, '%.2f'%(im[x, y]), fontsize=font_size)
    plt.savefig(name + '.jpg')

def get_eq_classes_of(y, points_per_class, nclass=5):
    y_ind = np.arange(y.shape[0])
    indices = np.asarray([y_ind[y[:, cl] == 1.][:points_per_class] for cl in range(nclass)]).reshape(nclass*points_per_class)
    return indices
    #return x_eq, y_eq

def get_tsne_data(X):

    pkled_data_name = "%s-%s-%s-%s-%i-%f-%i"%(pp_key, data_type,'-'.join(event_types),
                                              os.path.splitext(os.path.basename(filepath)), final_dim, perp, max_iter)
    new_pkled_data_name = "%s-%s-%s-%s-%i-%f-%i"%(pp_key, data_type,'-'.join(event_types),
                                              os.path.splitext(os.path.basename(filepath))[0], final_dim, perp, max_iter)
    full_path = os.path.join('pkled_tsne', pkled_data_name)
    new_full_path = os.path.join('pkled_tsne', new_pkled_data_name)
    if os.path.exists(full_path):
        print "reconstructing tsne for %s from pickled file. This shouldn't take long..."%(pkled_data_name)
        x_ts = pickle.load(open(full_path))
        pickle.dump(x_ts,open(new_full_path, 'w'))
        os.remove(full_path)
    elif os.path.exists(new_full_path):
        print "reconstructing tsne for %s from pickled file. This shouldn't take long..."%(pkled_data_name)
        x_ts = pickle.load(open(new_full_path))
    else:
        x_ts = tsne.tsne(X.astype('float64'), final_dim, X.shape[1], perp, max_iter=max_iter) #, 2, 10, 10.0
        pickle.dump(x_ts, open(new_full_path, 'w'))

    return x_ts

def save_images_close_to_centroids(x_ts, x_raw_eq):
         #find centroid of each class
        centroids = np.asarray([np.mean(x_ts[[y[:, cl] == 1.]], axis=0) for cl in range(nclass)])
        class_arr = np.asarray([x_ts[y[:, cl] == 1.] for cl in range(nclass)])

        #find closest t-sne points for each class for each centroid of the classes, so
        diff = np.asarray([[arr - cent for cent in centroids]for arr in class_arr])**2
        ix = np.argmin(diff[:, :, :, 0] + diff[:, :, :, 1], axis=2)


        #find the corresponding raw data to these closest points, so xij is the image of class i that is closest to the cetnroid for class j
        im_matrix = np.asarray([x_raw_eq[points_per_class*n+i, :] for n, i in enumerate(ix)])
        for cl in range(nclass):
            for cent in range(nclass):
                    im_name = 'im_of_class_%s_closes_to_centroid_of_class_%s_%s_%s'%(event_dict[cl], event_dict[cent], pp_key, data_type)
                    save_dir = im_path
                    for ext in [pp_key, data_type, '%s_close_to_cent_of_%s'%(event_dict[cl], event_dict[cent]) ]:
                        save_dir = os.path.join(save_dir,ext)
                        if not os.path.exists(save_dir):
                            os.mkdir(save_dir)
                    save_image(im_matrix[cl, cent], os.path.join(save_dir, im_name))





if len(sys.argv) > 1:
    filepath = sys.argv[1]
else:
    filepath='./results/192-284-284-10-Tanh-single_20000-rot-100-final.h5'

im_path="./images"
if not os.path.exists(im_path):
    os.mkdir(im_path)
final_dim = 2
perp = 50.0
max_iter = 500
base_path='./'
points_per_class=350
nclass = 5
ignore_muon=False
# file_name = 'single_20000'

#keep in this list in the order commented above
event_types = ['adinit', 'addelay', 'muo', 'fla', 'oth', ]
event_dict = {i: ev for i, ev in enumerate(event_types)}

post_process_types = ['ae', 'raw']
data_types = ['val']

h5file = h5py.File(filepath)


for data_type in data_types:
    for pp_key in post_process_types:
        x_pp = np.asarray(h5file[data_type + '_' + pp_key + '_x'])
        x_raw = np.asarray(h5file[data_type + '_' + 'raw' + '_x'])
        y = np.asarray(h5file[data_type + '_raw_y'])

        indices = get_eq_classes_of(y, points_per_class)
        X = x_pp[indices]
        x_raw_eq = x_raw[indices]
        y=y[indices]

        x_ts = get_tsne_data(X)

        save_images_close_to_centroids(x_ts, x_raw_eq)


        #pca = PCA(n_components=final_dim)
        x_pca = tsne.pca(X, final_dim)


        colors = ['b', 'r', 'g', 'y', 'm']
        markers = ['o', 'o', 'o', 's', 's']
        kwargs={}
        if final_dim == 3:
            kwargs['projection']='3d'
        reduced_arrs = {'tsne': x_ts, 'pca':x_pca}

        for red_type, x_red in reduced_arrs.iteritems():
            fig = plt.figure(1)
            plt.clf()
            ax = fig.add_subplot(111, **kwargs)

            #event_types list is in order of which bit is hot, where the most significant bit is first
            for hot_i, event in enumerate(event_types):
                if ignore_muon:
                    if event == 'muo':
                        continue

                x_red_ev = x_red[y[:, hot_i] == 1.]
                if final_dim == 3:
                    ax.scatter(x_red_ev[:, 0], x_red_ev[:, 1], x_red_ev[:, 2], marker=markers[hot_i], c=colors[hot_i], alpha=0.9, label=event)
                else:
                    ax.scatter(x_red_ev[:, 0], x_red_ev[:, 1], marker=markers[hot_i], c=colors[hot_i], alpha=0.9, label=event)

            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])


            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                      fancybox=True, shadow=True, ncol=5, prop={'size': 6})
            if not os.path.exists('./plots'):
                os.mkdir('./plots')
            plt.savefig('./plots/%s%s-%s-%s-%iD-%s.pdf' % (('no-muo-' if ignore_muon else ''), red_type, pp_key, data_type, final_dim, os.path.splitext(os.path.basename(filepath))[0]))

