import time

t1 = time.time()
import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot as plt

import numpy as np
import h5py

import os

from tsne_source_code import tsne
import pickle
import itertools

# from mpl_toolkits.mplot3d import Axes3D
import sys
from util.helper_fxns import get_eq_classes_of
#self.base_path='/scratch3/scratchdirs/jialin/dayabay/'


# 1) Primary AD           10000
# 2) Delayed AD response  01000
# 3) Muon decay           00100
# 4) Flasher              00010
# 5) Other (background noise) 00001



class Vis(object):
    def __init__(self, filepath='./results/192-284-284-10-Tanh-single_20000-rot-100-final.h5', reconstruct=True, old=False, ignore='', plot_tsne=False, highlight_centroid=True, perplexity=50.0, max_iter=500):
        self.plot_tsne = plot_tsne
        self.filepath = filepath
        self.highlight_centroid = highlight_centroid
        self.raw_dim = 192
        self.im_path = "./images"
        self.old = old
        if not os.path.exists(self.im_path):
            os.mkdir(self.im_path)
        self.final_dim = 2
        self.perp = perplexity
        self.max_iter = max_iter
        self.base_path = './'
        self.cent_images_per_class = 5
        self.tsne_points_per_class = 500
        self.plot_points_per_class = 500
        self.nclass = 5
        self.ignore = ignore.split(',')

        self.reconstruct=reconstruct
        # file_name = 'single_20000'

        #keep in this list in the order commented above
        self.event_types = ['ibd_prompt', 'ibd_delay', 'muon', 'flasher', 'other' ]
        self.event_dict = {i: ev for i, ev in enumerate(self.event_types)}

        self.post_process_types = ['conv-ae']
        self.data_types = ['val']
        self.pp_type = None
        self.data_type = None
        self.h5file = h5py.File(self.filepath)


    def save_image(self, vec, name, y_offset=0.3, x_offset=0.69, font_size=5):
        plt.clf()
        im = vec.reshape(8, 24)
        ax = plt.imshow(im, interpolation='none')
        left, right, bottom, top = ax.get_extent()
        plt.colorbar(orientation="horizontal")
        for x in range(im.shape[0]):
            for y in range(im.shape[1]):
                plt.text(left + y_offset + y, top + x_offset + x, '%.2f' % (im[x, y]), fontsize=font_size)
        plt.savefig(name + '.jpg')



    def get_tsne_data(self, X, data_type, pp_type):
        ts_key = data_type + '/' + pp_type + '/' + 'ts_' + data_type + '_' + pp_type + '_x'
        calc=True
        if self.reconstruct:
            if ts_key in self.h5file:
                x_ts = np.asarray(self.h5file[ts_key])
                calc = False
        if calc:
            x_ts = tsne.tsne(X.astype('float64'), self.final_dim, X.shape[1], self.perp,
                                 max_iter=self.max_iter)
            self.h5file.create_dataset(ts_key, data=x_ts)
        return x_ts

    def get_centroids(self, x, y):
        centroids = np.asarray([np.mean(x[[y[:, cl] == 1.]], axis=0) for cl in range(self.nclass)])
        return centroids

    def save_images_close_to_centroids(self, y, x, x_raw_eq, pp_name):
        #find centroid of each class
        centroids = self.get_centroids(x,y)
        class_arr = np.asarray([x[y[:, cl] == 1.] for cl in range(self.nclass)])

        #find closest t-sne points for each class for each centroid of the classes, so
        diff = np.asarray([[arr - cent for cent in centroids] for arr in class_arr]) ** 2

        ix = np.argmin(diff[:, :, :, 0] + diff[:, :, :, 1], axis=2)
        ix = np.argsort(diff[:, :, :, 0] + diff[:, :, :, 1], axis=2)[:,:,:self.cent_images_per_class]


        #find the corresponding raw data to these closest points, so xij is the image of class i that is closest to the cetnroid for class j
        im_matrix = np.asarray([x_raw_eq[self.tsne_points_per_class * n + i, :] for n, i in enumerate(ix)])
        for cl in range(self.nclass):
            for cent in range(self.nclass):
                save_dir = self.im_path
                for ext in [self.pp_type, self.data_type,'%s_close_to_cent_of_%s' % (self.event_dict[cl], self.event_dict[cent])]:
                    save_dir = os.path.join(save_dir, ext)
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                for im_no in range(self.cent_images_per_class):
                    im_name = '%s-im_of_class_%s_closes_to_centroid_of_class_%s_%s_%s_%i' % (
                        pp_name,self.event_dict[cl], self.event_dict[cent], self.pp_type, self.data_type, im_no)
                    self.save_image(im_matrix[cl, cent,im_no], os.path.join(save_dir, im_name))


    def generate_save_path(self, red_type, ignores, data_type, pp_type):
        save_path = './plots/%s-%s-%s/%s-%iD-%s-%s-%s-%s-%s.pdf' % (
                        red_type,
                        pp_type,
                        data_type,
                        red_type,
                        self.final_dim,
                        '-'.join(['no_' + ig for ig in ignores]),
                        pp_type,
                        data_type,
                        str(self.perp) if red_type == 't-SNE' else '',
                        os.path.splitext(os.path.basename(self.filepath))[0])
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        return save_path

    def get_data(self, data_type, pp_type):
        if self.old:
            x_pp = np.asarray(self.h5file[data_type + '_' + pp_type + '_x'])
            x_raw = np.asarray(self.h5file[data_type + '_' + 'raw' + '_x'])
            y = np.asarray(self.h5file[data_type + '_raw_y'])
        else:

            x_pp = np.asarray(self.h5file[pp_type + '/' + data_type + '/x'])
            x_raw = np.asarray(self.h5file['raw' + '/' + data_type + '/x'])
            y = np.asarray(self.h5file['raw' + '/' + data_type + '/y'])
        return x_pp, x_raw, y


    def _plot(self, x_red, red_type, y, ignores, data_type,pp_type):
        colors = ['b', 'r', 'g', 'y', 'm']
        kwargs = {}
        if self.final_dim == 3:
            kwargs['projection'] = '3d'

        fig = plt.figure(1)
        plt.clf()
        ax = fig.add_subplot(111, **kwargs)
        centroids = self.get_centroids(x_red,y)
        for hot_i, event in enumerate(self.event_types):
            if event in ignores:
                continue
            x_red_ev = x_red[y[:, hot_i] == 1.][:self.plot_points_per_class]
            if self.final_dim == 3:
                ax.scatter(x_red_ev[:, 0], x_red_ev[:, 1], x_red_ev[:, 2], marker='o', c=colors[hot_i],
                           alpha=0.9, label=event)
            else:
                ax.scatter(x_red_ev[:, 0], x_red_ev[:, 1], marker='o', edgecolors=colors[hot_i], c=colors[hot_i], alpha=0.1,
                           label=event)

                if self.highlight_centroid:
                    ax.scatter(centroids[hot_i, 0], centroids[hot_i, 1], edgecolors='black',  marker='s', c=colors[hot_i])

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, shadow=True, ncol=5, prop={'size': 12})

        save_path = self.generate_save_path(red_type, ignores, data_type, pp_type)
        plt.savefig(save_path)


    def plot(self):
        for data_type in self.data_types:
            for pp_type in self.post_process_types:

                x_pp, x_raw, y = self.get_data(data_type, pp_type)
                indices = get_eq_classes_of(y, self.tsne_points_per_class, self.nclass)
                x_eq, x_raw_eq, y_eq = map(lambda c : c[indices], [x_pp, x_raw, y])

                if self.plot_tsne:
                    x_ts = self.get_tsne_data(x_eq, pp_type, data_type)
                    self._plot(x_ts,'t-SNE', y_eq, self.ignore)

                x_pca = tsne.pca(x_eq, self.final_dim)
                self.h5file.create_dataset(pp_type + '/' + data_type + '/' + 'x_pca', data=x_pca)
                self._plot(x_pca, 'PCA', y_eq, self.ignore, data_type, pp_type)
                #self.save_images_close_to_centroids(y, x_pca, x_raw)



if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = './results/old/192-284-284-10-Tanh-single_20000-rot-100-final.h5'

    pl = Vis(filepath, reconstruct=False, old=True, ignore='other,flasher,ibd_delay')
    pl.plot()



