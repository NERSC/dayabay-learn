
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
#self.base_path='/scratch3/scratchdirs/jialin/dayabay/'


# 1) Primary AD           10000
# 2) Delayed AD response  01000
# 3) Muon decay           00100
# 4) Flasher              00010
# 5) Other (background noise) 00001

from matplotlib import pyplot as plt
class TsneVis(object):

    
    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_dim = 192
        self.im_path = "./images"
        if not os.path.exists(self.im_path):
            os.mkdir(self.im_path)
        self.final_dim = 2
        self.perp = 50.0
        self.max_iter = 500
        self.base_path = './'
        self.tsne_points_per_class = 350
        self.plot_points_per_class = 350
        self.nclass = 5
        self.ignore_muon = False
        self.ignore_flasher = False
        # file_name = 'single_20000'

        #keep in this list in the order commented above
        self.event_types = ['adinit', 'addelay', 'muo', 'fla', 'oth', ]
        self.event_dict = {i: ev for i, ev in enumerate(self.event_types)}

        self.post_process_types = ['ae', 'raw']
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
                plt.text(left + y_offset + y, top + x_offset + x, '%.2f'%(im[x, y]), fontsize=font_size)
        plt.savefig(name + '.jpg')

    def get_eq_classes_of(self, y):
        y_ind = np.arange(y.shape[0])
        indices = np.asarray([y_ind[y[:, cl] == 1.][:self.tsne_points_per_class] for cl in range(self.nclass)]).reshape(self.nclass*self.tsne_points_per_class)
        return indices
        #return x_eq, y_eq

    def get_tsne_data(self,X):

        pkled_data_name = "%s-%s-%s-%s-%i-%f-%i"%(self.pp_type, self.data_type,'-'.join(self.event_types),
                                                  os.path.splitext(os.path.basename(self.filepath)), self.final_dim,
                                                  self.perp,
                                                  self.max_iter)
        new_pkled_data_name = "%s-%s-%s-%s-%i-%f-%i"%(self.pp_type,
                                                      self.data_type,'-'.join(self.event_types),
                                                  os.path.splitext(os.path.basename(self.filepath))[0],
                                                      self.final_dim,
                                                      self.perp,
                                                      self.max_iter)
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
            x_ts = tsne.tsne(X.astype('float64'), self.final_dim, X.shape[1], self.perp, max_iter=self.max_iter) #, 2, 10, 10.0
            pickle.dump(x_ts, open(new_full_path, 'w'))

        return x_ts

    def save_images_close_to_centroids(self, y, x_ts, x_raw_eq):
             #find centroid of each class
            centroids = np.asarray([np.mean(x_ts[[y[:, cl] == 1.]], axis=0) for cl in range(self.nclass)])
            class_arr = np.asarray([x_ts[y[:, cl] == 1.] for cl in range(self.nclass)])

            #find closest t-sne points for each class for each centroid of the classes, so
            diff = np.asarray([[arr - cent for cent in centroids]for arr in class_arr])**2
            ix = np.argmin(diff[:, :, :, 0] + diff[:, :, :, 1], axis=2)


            #find the corresponding raw data to these closest points, so xij is the image of class i that is closest to the cetnroid for class j
            im_matrix = np.asarray([x_raw_eq[self.tsne_points_per_class*n+i, :] for n, i in enumerate(ix)])
            for cl in range(self.nclass):
                for cent in range(self.nclass):
                        im_name = 'im_of_class_%s_closes_to_centroid_of_class_%s_%s_%s'%(self.event_dict[cl], self.event_dict[cent], self.pp_type, self.data_type)
                        save_dir = self.im_path
                        for ext in [self.pp_type, self.data_type, '%s_close_to_cent_of_%s'%(self.event_dict[cl], self.event_dict[cent]) ]:
                            save_dir = os.path.join(save_dir,ext)
                            if not os.path.exists(save_dir):
                                os.mkdir(save_dir)
                        self.save_image(im_matrix[cl, cent], os.path.join(save_dir, im_name))
    def _plot(self,reduced_arrs, y):
        colors = ['b', 'r', 'g', 'y', 'm']
        markers = ['o', 'o', 'o', 's', 's']
        kwargs={}
        if self.final_dim == 3:
            kwargs['projection']='3d'


        for red_type, x_red in reduced_arrs.iteritems():
            fig = plt.figure(1)
            plt.clf()
            ax = fig.add_subplot(111, **kwargs)

            #self.event_types list is in order of which bit is hot, where the most significant bit is first
            for hot_i, event in enumerate(self.event_types):
                if self.ignore_muon:
                    if event == 'muo':
                        continue
                if self.ignore_flasher:
                    if event == 'fla':
                        continue

                x_red_ev = x_red[y[:, hot_i] == 1.][:self.plot_points_per_class]
                if self.final_dim == 3:
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
            plt.savefig('./plots/%s%s%s-%s-%s-%iD-%s.pdf' %
                        (('no-muo-' if self.ignore_muon else ''),('no-fla-' if self.ignore_flasher else '') ,
                         red_type,
                         self.pp_type,
                         self.data_type,
                         self.final_dim,
                         os.path.splitext(os.path.basename(self.filepath))[0]))


    def plot_tsne(self):
        for data_type in self.data_types:
            self.data_type = data_type
            for pp_type in self.post_process_types:

                self.pp_type = pp_type
                
                x_pp = np.asarray(self.h5file[self.data_type + '_' + self.pp_type + '_x'])
                x_raw = np.asarray(self.h5file[self.data_type + '_' + 'raw' + '_x'])
                y = np.asarray(self.h5file[self.data_type + '_raw_y'])

                indices = self.get_eq_classes_of(y)
                X = x_pp[indices]
                x_raw_eq = x_raw[indices]
                y=y[indices]

                x_ts = self.get_tsne_data(X)

                self.save_images_close_to_centroids(y,x_ts, x_raw_eq)


                #pca = PCA(n_components=self.final_dim)
                x_pca = tsne.pca(X, self.final_dim)
                reduced_arrs = {'tsne': x_ts, 'pca':x_pca}
                self._plot(reduced_arrs, y)












if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath='./results/192-284-284-10-Tanh-single_20000-rot-100-final.h5'

    plt_tsne = TsneVis(filepath)
    plt_tsne.plot_tsne()



