__author__ = 'racah'
import h5py
import sys
import os
import numpy as np
from util.helper_fxns import get_eq_classes_of
from matplotlib import pyplot as plt

def plot_nums(ax, y_offset=0.3, x_offset=0.69, font_size=5):
    left, right, bottom, top = ax.get_extent()
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            ax.text(left + y_offset + y, top + x_offset + x, '%.2f' % (im[x, y]), fontsize=font_size)
    return ax


event_types = ['ibd_prompt', 'ibd_delay', 'muon', 'flasher', 'other' ]
event_dict = {i: ev for i, ev in enumerate(event_types)}

#h5file_path = sys.argv[1]
h5file_path = './results/final_results_tr_size_31700200.0005.h5'
h5f = h5py.File(h5file_path)
reconstr_val = np.asarray(h5f['conv-ae/val/x_reconstructed'])
init_val_x = np.asarray(h5f['raw/val/x'])
val_y = np.asarray(h5f['raw/val/y'])

indices = get_eq_classes_of(val_y, points_per_class=5, nclass=5).reshape(5,5)

raw_d = dict(zip(event_types,[ init_val_x[i] for i in indices]))
rec_d = dict(zip(event_types,[ reconstr_val[i] for i in indices]))

y_offset = 0.1
x_offset= 0.2
font_size = 3
for key in raw_d.keys():
    for i, im in enumerate(raw_d[key]):
        plt.figure(i)
        plt.clf()
        raw_s = plt.subplot(2, 1, 1)
        rec_s = plt.subplot(2, 1, 2)
        raw_i = raw_d[key][i].reshape(8,24)
        rec_i = rec_d[key][i].reshape(8,24)
        raw_s.imshow(raw_i, interpolation='none')
        rec_s.imshow(rec_i, interpolation='none')
        plt.title('%s event reconstructed' % (key))
        #plt.colorbar(orientation="horizontal")
        #rec_s.colorbar(orientation="horizontal")
        # rec_s = plot_nums(rec_s, y_offset=0.3, x_offset=0.69, font_size=5)
        # raw_s = plot_nums(raw_s, y_offset=0.3, x_offset=0.69, font_size=5)
        if not os.path.exists('./images/reconstructed/'):
            os.mkdir('./images/reconstructed/')
        plt.savefig('./images/reconstructed/' + key + str(i) + '.jpg')




#
#
# for i in range(len(val_i)):
#     v_i = val_i[i]
#     plt.figure(1)
#     plt.clf()
#     pred = plt.subplot(3,1,1)
#     pred.imshow(prob_map[i])
#     hur_ch = plt.subplot(3,1,2)
#     hur_ch.imshow(cropped_ims[v_i, 2, :, :])
#     hur_ch.add_patch(patches.Rectangle(
#         (boxes[v_i][0, 0] - rad, boxes[v_i][0, 1] - rad),
#         boxes[v_i][0, 2] - boxes[v_i][0, 0],
#         boxes[v_i][0, 3] - boxes[v_i][0, 1],
#         fill=False))
#     # pred.add_patch(patches.Rectangle(
#     #     (boxes[v_i][0, 0] - rad, boxes[v_i][0, 1] - rad),
#     #     boxes[v_i][0, 2] - boxes[v_i][0, 0],
#     #     boxes[v_i][0, 3] - boxes[v_i][0, 1],
#     #     fill=False))
#     gr_truth = plt.subplot(3,1,3)
#     x=cropped_ims[0].shape[1]
#     y=cropped_ims[0].shape[2]
#     gr_truth.imshow(y_val[i*x*y : (i+1)*x*y].reshape(x,y))
#
#     plt.savefig(os.path.join(dirs['images_dir'], '%s-%i.pdf' % (model_key, i)))