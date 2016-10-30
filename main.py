""" 

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#optional-install-cuda-gpus-on-linux
http://tflearn.org/installation/
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0-cp35-cp35m-linux_x86_64.whl


export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-7.5/targets/x86_64-linux/lib"
export CUDA_HOME=/usr/local/cuda


"""
from __future__ import division, print_function, absolute_import

import os
import sys

sys.path.append('/mnt/dokumneter/sintef/NeuroImageRegistration/')
import sqlite3
import nibabel as nib
import numpy as np
import h5py

import util

# Data loading and preprocessing
# import tflearn.datasets.oxflower17 as oxflower17
# X, Y = oxflower17.load_data(one_hot=True)


util.setup("GBM_deepNN/")
if not os.path.exists(util.TEMP_FOLDER_PATH):
    os.makedirs(util.TEMP_FOLDER_PATH)

conn = sqlite3.connect(util.DB_PATH)
conn.text_factory = str
cursor = conn.execute('''SELECT pid from QualityOfLife''')


def get_data():
    filename = 'gbm_data.h5'
    if not os.path.exists(filename):
        X = []
        Y = []
        PID = []
        conn = sqlite3.connect(util.DB_PATH)
        conn.text_factory = str

        image_ids = util.find_images_with_qol()
        print(image_ids)
        i = 0
        for image_id in image_ids:
            print(image_id)
            cursor = conn.execute('''SELECT filepath_reg, pid from Images where id = ? ''', (image_id,))
            db_temp = cursor.fetchone()
            if not db_temp:
                continue
            t1_vol = util.DATA_FOLDER + db_temp[0]
            pid = db_temp[1]
            qol = conn.execute('SELECT {} from QualityOfLife where pid = ?'.format('Global_index'), (pid,)).fetchone()[
                0]
            if not qol:
                continue

            (label_vol, label) = util.find_reg_label_images(image_id)[0]
            label_img = nib.load(label_vol)
            label_data = np.array(label_img.get_data())
            temp = np.sum(np.sum(label_data, axis=0), axis=0)
            z = np.argmax(temp)
            label_slice = label_data[:, :, z]
            # np.save(util.TEMP_FOLDER_PATH + "/res/" + pid + "_label", label_slice)

            t1_img = nib.load(t1_vol)
            t1_data = np.array(t1_img.get_data())
            # t1_slice = t1_data[:, :, z]
            # np.save(util.TEMP_FOLDER_PATH + "/res/" + pid + "_t1", t1_slice)

            # print(t1_slice[:, np.newaxis].shape)
            X = t1_data[:, :, :, np.newaxis]
            if not os.path.exists(filename):
                # Open a file in "w"rite mode
                d_imgshape = (len(image_ids), X.shape[0], X.shape[1], X.shape[2], X.shape[3])
                d_labelshape = (len(image_ids),)

                dataset = h5py.File(filename, 'w')
                dataset.create_dataset('X', d_imgshape, chunks=True)
                dataset.create_dataset('Y', d_labelshape, chunks=True)
                #            dataset.close()

                #        dataset = h5py.File(filename, 'w')
            dataset['X'][i] = X
            dataset['Y'][i] = (qol - 1)
            dataset.flush()
            i = i + 1
            print(temp.shape)

        cursor.close()
        conn.close()
        dataset.close()

    h5f = h5py.File(filename, 'r')
    return h5f


def get_data2():
    filename = 'gbm_data2.h5'
    if not os.path.exists(filename):
        X = []
        Y = []
        PID = []
        conn = sqlite3.connect(util.DB_PATH)
        conn.text_factory = str

        cursor2 = conn.execute('''SELECT id,filepath_reg from Images where diag_pre_post = ?''',("pre",))
        image_ids = []
        for _id in cursor2:
            if _id[1] is None:
                continue
            image_ids.append(_id[0])
        cursor2.close()

        i = 0
        for image_id in image_ids:
            print(image_id)
            cursor = conn.execute('''SELECT filepath_reg, pid from Images where id = ? ''', (image_id,))
            db_temp = cursor.fetchone()
            if not db_temp:
                continue
            t1_vol = util.DATA_FOLDER + db_temp[0]
            pid = db_temp[1]
            glioma_grade = conn.execute('''SELECT glioma_grade from Patient where pid = ?''', (pid,)).fetchone()[
                0]
            if not glioma_grade:
                continue

            (label_vol, label) = util.find_reg_label_images(image_id)[0]
            label_img = nib.load(label_vol)
            label_data = np.array(label_img.get_data())
            temp = np.sum(np.sum(label_data, axis=0), axis=0)
            z = np.argmax(temp)
            label_slice = label_data[:, :, z]
            # np.save(util.TEMP_FOLDER_PATH + "/res/" + pid + "_label", label_slice)

            t1_img = nib.load(t1_vol)
            t1_data = np.array(t1_img.get_data())
            # t1_slice = t1_data[:, :, z]
            # np.save(util.TEMP_FOLDER_PATH + "/res/" + pid + "_t1", t1_slice)

            # print(t1_slice[:, np.newaxis].shape)
            X = t1_data[:, :, :, np.newaxis]
            if not os.path.exists(filename):
                # Open a file in "w"rite mode
                d_imgshape = (len(image_ids), X.shape[0], X.shape[1], X.shape[2], X.shape[3])
                d_labelshape = (len(image_ids),)

                dataset = h5py.File(filename, 'w')
                dataset.create_dataset('X', d_imgshape, chunks=True)
                dataset.create_dataset('Y', d_labelshape, chunks=True)
                #            dataset.close()

                #        dataset = h5py.File(filename, 'w')
            dataset['X'][i] = X
            dataset['Y'][i] = glioma_grade - 2
            dataset.flush()
            i = i + 1
            print(temp.shape)

        cursor.close()
        conn.close()
        dataset.close()

    h5f = h5py.File(filename, 'r')
    return h5f

h5f = get_data()
X = h5f['X']
Y = h5f['Y']

#X_test = h5f['cifar10_X_test']

print(min(Y), max(Y))

#Y_test = h5f['cifar10_Y_test']

# X_test = X[100:]
# Y_test = Y[100:]
# X = X[:100]
# Y = Y[:100]

import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_3d, max_pool_3d, avg_pool_3d
from tflearn.layers.estimator import regression

Y = to_categorical(Y, int(max(Y)+1))

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Building 'VGG Network'
network = input_data(shape=[None, 197, 233, 189, 1])

network = conv_3d(network, 4, 3, activation='relu')
network = conv_3d(network, 4, 3, activation='relu')
network = max_pool_3d(network, [1, 2, 2, 2, 1], strides=2)

network = conv_3d(network, 8, 3, activation='relu')
network = conv_3d(network, 8, 3, activation='relu')
network = max_pool_3d(network, [1, 2, 2, 2, 1], strides=2)

network = conv_3d(network, 64, 3, activation='relu')
network = conv_3d(network, 64, 3, activation='relu')
network = max_pool_3d(network, [1, 2, 2, 2, 1], strides=2)

network = conv_3d(network, 128, 3, activation='relu')
network = conv_3d(network, 128, 3, activation='relu')
network = max_pool_3d(network, [1, 2, 2, 2, 1], strides=2)

network = conv_3d(network, 256, 3, activation='relu')
network = conv_3d(network, 256, 3, activation='relu')
network = conv_3d(network, 256, 3, activation='relu')
network = max_pool_3d(network, [1, 2, 2, 2, 1], strides=2)

network = conv_3d(network, 512, 3, activation='relu')
network = conv_3d(network, 512, 3, activation='relu')
network = conv_3d(network, 512, 3, activation='relu')
network = max_pool_3d(network, [1, 2, 2, 2, 1], strides=2)

network = conv_3d(network, 512, 3, activation='relu')
network = conv_3d(network, 512, 3, activation='relu')
network = conv_3d(network, 512, 3, activation='relu')
network = max_pool_3d(network, [1, 2, 2, 2, 1], strides=2)

network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 3, activation='softmax')

network = regression(network, optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=500, shuffle=True,
          show_metric=True, batch_size=32, snapshot_step=500,
          snapshot_epoch=False, run_id='gbm_qpø')

h5f.close()
