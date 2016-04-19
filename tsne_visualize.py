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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#self.base_path='/scratch3/scratchdirs/jialin/dayabay/'


# 1) Primary AD           10000
# 2) Delayed AD response  01000
# 3) Muon decay           00100
# 4) Flasher              00010
# 5) Other (background noise) 00001



class Vis(object):
    def __init__(self, filepath='./results/192-284-284-10-Tanh-single_20000-rot-100-final.h5', reconstruct=True, pp_types='conv-ae', data_types='val',
                 old=False, ignore='', plot_tsne=False, plot_pca=True, highlight_centroid=True, perplexity=50.0, max_iter=500):
        self.plot_tsne = plot_tsne
        self.filepath = filepath
        self.highlight_centroid = highlight_centroid
        self.raw_dim = 192
        self.im_path = "./images"
        self.old = old
        self.plot_pca = plot_pca
        if not os.path.exists(self.im_path):
            os.mkdir(self.im_path)
        self.final_dim = 2
        self.perp = perplexity
        self.max_iter = max_iter
        self.base_path = './'
        self.cent_images_per_class = 3
        self.tsne_points_per_class = 500
        self.plot_points_per_class = 500
        self.nclass = 5
        self.ignore = ignore.split(',')

        self.reconstruct=reconstruct
        # file_name = 'single_20000'

        #keep in this list in the order commented above
        self.event_types = ['ibd_prompt', 'ibd_delay', 'muon', 'flasher', 'other' ]
        self.event_dict = {i: ev for i, ev in enumerate(self.event_types)}

        self.post_process_types = pp_types.split(',') #['conv-ae']
        self.data_types = data_types.split(',') #['val']
        self.pp_type = None
        self.data_type = None
        self.h5file = h5py.File(self.filepath)


    def __del__(self):
        self.h5file.close()

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

    def save_z_ims(self,plot_name, z, path, *args):
        plt.figure(1)
        plt.clf()
        for i in range(z):
            sp = plt.subplot(z,1,i + 1)
            im = args[i].reshape(8,24)
            sp.imshow(im,interpolation='none' )

        #plt.title(plot_name)
        #plt.colorbar(orientation="horizontal")

	self._mkdir_recursive(path)
        plt.savefig(path + plot_name.replace(' ', '_') + '.jpg')


    def create_key(self,data_type, pp_type, red_type ):
        return pp_type + '/' + data_type + '/' + 'x_' + red_type

    def reconstruct_data(self, data_type, pp_type, red_type):
        key = self.create_key(data_type, pp_type, red_type)
        calc = True
        x= None
        if self.reconstruct:
            if key in self.h5file:
                print "reconstructing " + pp_type + " data..."
                x = np.asarray(self.h5file[key])

                calc = False
        return calc, x
    #Thanks, Mars from stackoverflow!
    def _mkdir_recursive(self, path):
    	sub_path = os.path.dirname(path)
    	if not os.path.exists(sub_path):
        	self._mkdir_recursive(sub_path)
    	if not os.path.exists(path):
        	os.mkdir(path)

    def get_pca_data(self, X, data_type, pp_type):
        calc, x_pca = self.reconstruct_data(data_type, pp_type, 'pca')
        if calc:
            print 'calculating pca for %s-%s'%(data_type, pp_type)
            x_pca = PCA(self.final_dim).fit_transform(X)
            self.h5file.create_dataset(pp_type + '/' + data_type + '/' + 'x_pca', data=x_pca)
        return x_pca

    def get_tsne_data(self, X, data_type, pp_type):
        calc, x_ts = self.reconstruct_data(data_type, pp_type, 'ts')
        if calc:
            print "calclating tsne... "
            #x_ts = tsne.tsne(X.astype('float64'), self.final_dim, X.shape[1], self.perp,
            #                   max_iter=self.max_iter)
	    ts = TSNE(n_components=self.final_dim,perplexity=self.perp,n_iter=self.max_iter)
	    x_ts = ts.fit_transform(X.astype('float64'))
            key = self.create_key(data_type, pp_type, 'ts')
            if key in self.h5file:
                self.h5file.__delitem__(key)
            self.h5file.create_dataset(key, data=x_ts)
        return x_ts

    def get_event_num(self, event_name):
        return self.event_types.index(event_name)

    def get_array_of_all_event(self,x,y,event_name):
        num = self.get_event_num(event_name)

        return x[y[:, num] == 1.]

    def get_centroid(self, x, y, label_num):
        return np.mean(x[[y[:, label_num] == 1.]], axis=0)

    def get_j_closest_to_centroid_of_k(self,x,y,event_j,event_k, num_points):
        j_num = self.get_event_num(event_j)
        k_num = self.get_event_num(event_k)
        cent = self.get_centroid(x,y, label_num=k_num)
        event_j_ex = self.get_array_of_all_event(x,y, event_j)
        l2_dist = np.sum((event_j_ex - cent) **2, axis = 1)

        #returns index to array of all j's (so to use ix_j, you must do (x[y[:, j_num]] == 1)[ix_j]
        ix_j = np.argsort(l2_dist, axis =0)[:num_points]
        return ix_j



    def save_image_j_close_to_centroid_k(self, y, x, x_raw,j, k, pp_type, x_rec=None):

        event = j
        cent = k
        event_num = self.get_event_num(event)
        ixs = self.get_j_closest_to_centroid_of_k(x,y,event, cent, self.cent_images_per_class)
        raw_ims = self.get_array_of_all_event(x_raw, y, event)
        i = 0
        ix = ixs[i]
        raw_im = raw_ims[ix]
        name = '%s_%s_' %(event, cent) + str(ix)
        if x_rec is not None:
            rec_ims = self.get_array_of_all_event(x_rec, y, event)
            rec_im = rec_ims[ix]
            self.save_z_ims(name, 2, 'images/reconstructed/' + pp_type + '/', raw_im, rec_im)
        else:
            self.save_image(raw_im,name)

        #returns index into array of just one class to get ith image
        return ix, i, name


    def generate_save_path(self, red_type, ignores, data_type, pp_type, with_annot=False):
        save_path = './plots/%s-%s-%s/%s-%iD-%s-%s-%s-%s-%s-%s.jpg' % (
                        red_type,
                        pp_type,
                        data_type,
                        red_type,
                        self.final_dim,
                        '-'.join(['no_' + ig for ig in ignores]),
                        pp_type,
                        data_type,
                        str(self.perp) if red_type == 't-SNE' else '',
                        "annot" if with_annot else '',
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


    def _plot(self, x_red, red_type, y, ignores, data_type,pp_type, x_raw, x_rec=None, with_annot=False):
        d={}
        for event in self.event_types:
            for cent in self.event_types:

                    ix, i, name = self.save_image_j_close_to_centroid_k( y, x_red, x_raw, event, cent, pp_type, x_rec=x_rec)
                    d[event + '_' + cent] = (ix, name)
        colors = ['b', 'r', 'g', 'y', 'm']
        kwargs = {}
        if self.final_dim == 3:
            kwargs['projection'] = '3d'

        fig = plt.figure(1)
        plt.clf()
        ax = fig.add_subplot(111, **kwargs)

        for hot_i, event in enumerate(self.event_types):
            if event in ignores:
                continue
            x_red_ev = x_red[y[:, hot_i] == 1.]#[:self.plot_points_per_class]
            if self.final_dim == 3:
                ax.scatter(x_red_ev[:, 0], x_red_ev[:, 1 ], x_red_ev[:, 2], marker='o', c=colors[hot_i],
                           alpha=0.9, label=event)
            else:
                ax.scatter(x_red_ev[:, 0], x_red_ev[:, 1], marker='o',  c=colors[hot_i], alpha=0.9,
                           label=event) #edgecolors=colors[hot_i],

                if self.highlight_centroid:
                    centroid = self.get_centroid(x_red, y, hot_i)
                    ax.scatter(centroid[0], centroid[1], edgecolors='black',  marker='s', c=colors[hot_i])

            if with_annot:
                for i,k in enumerate(d.keys()):
                    # if event == k.split('_')[0] and event == k.split('_')[1]:
                        ix, name = d[k]
                        #print x_red_ev[ix,0], x_red_ev[ix,1]
                        plt.annotate(name, xy= (x_red_ev[ix,0], x_red_ev[ix,1]),  xytext=(5*i,5*i),
                            textcoords = 'offset points', ha='right', va = 'bottom',
                            bbox = dict(boxstyle = 'round,pad=0.', fc = 'yellow', alpha = 0.5),
                            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'), fontsize=10)

        #
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.1,
        #                  box.width, box.height * 0.9])
        #
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
        #           fancybox=True, shadow=True, ncol=5, prop={'size': 12})

        #plt.show()
        save_path = self.generate_save_path(red_type, ignores, data_type, pp_type,with_annot)
        print save_path
        plt.savefig(save_path)


    def plot(self):
        for data_type in self.data_types:
            for pp_type in self.post_process_types:
                x_rec = self.h5file.get('conv-ae' + '/' + data_type + '/x_reconstructed', None)
                if x_rec:
                    x_rec = np.asarray(x_rec)
                x_pp, x_raw, y = self.get_data(data_type, pp_type)
                indices = get_eq_classes_of(y, self.tsne_points_per_class, self.nclass)
                x_eq, x_raw_eq, y_eq = map(lambda c : c[indices], [x_pp, x_raw, y])
                # if self.plot_pca:
                #     x_pca = self.get_pca_data(x_pp,data_type, pp_type)
                #     self._plot(x_pca, 'PCA', y, self.ignore, data_type, pp_type)

                if self.plot_tsne:
                    x_ts = self.get_tsne_data(x_pp, pp_type, data_type)
                    self._plot(x_ts,'t-SNE', y, self.ignore, data_type, pp_type, x_raw, x_rec)
                    self._plot(x_ts,'t-SNE', y, self.ignore, data_type, pp_type, x_raw, x_rec, with_annot=True)
    





if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = './results/final_results_tr_size_31700200.0005.h5'

    max_iter=500

    # pl = Vis(filepath, reconstruct=True, old=True)
    # pl.plot()
    v = Vis(filepath, old=False, plot_tsne=True,
            reconstruct=False, pp_types='conv-ae', data_types='val', max_iter=max_iter)
    v.plot()



