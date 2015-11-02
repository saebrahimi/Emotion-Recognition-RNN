#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import tables

convnet_datafile = '/home/vincent/data/emotiw2015/Datasets/convnet_data.h5'
mean_std_file = '/home/vincent/data/emotiw2015/Datasets/fertfd_mean_std.h5'

from data_provider import EmotionData

numpy_rng = np.random.RandomState(1)

fer_path = ('/home/vincent/data/emotiw2015/Datasets/registered48x48/'
            'fer2013_images_histeq')
tfd_path = ('/home/vincent/data/emotiw2015/Datasets/registered48x48/'
            'TFD_reg')
emotiw_train_path = (
    '/home/vincent/data/emotiw2015/Datasets/registered48x48/'
    'Emotiw15_Train_reg')
emotiw_val_path = (
    '/home/vincent/data/emotiw2015/Datasets/registered48x48/'
    'Emotiw15_Val_reg')

dataset = EmotionData(
    numpy_rng=numpy_rng, fer_path=fer_path, tfd_path=tfd_path,
    emotiw_train_path=emotiw_train_path, emotiw_val_path=emotiw_val_path,
    partitioning='convnet')

print 'computing FER+TFD mean and std images...'
mean0 = dataset.train_inputs.mean(0).astype(np.float32)
std0 = dataset.train_inputs.std(0).astype(np.float32)

with tables.openFile(mean_std_file, 'w') as h5file:
    h5file.createArray(h5file.root, 'mean0', mean0)
    h5file.createArray(h5file.root, 'std0', std0)

with tables.openFile(convnet_datafile, 'w') as h5file:
    h5file.createArray(h5file.root, 'train_inputs', dataset.train_inputs)
    h5file.createArray(h5file.root, 'val_inputs', dataset.val_inputs)
    h5file.createArray(h5file.root, 'train_outputs', dataset.train_outputs)
    h5file.createArray(h5file.root, 'val_outputs', dataset.val_outputs)

# vim: set ts=4 sw=4 sts=4 expandtab:
