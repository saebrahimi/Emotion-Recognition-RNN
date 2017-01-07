# coding: utf-8

import os

import numpy as np
import tables

from data_provider import EmotionData


# adapt the paths to match your folder structure
data_main_folder = '/home/usr/data/emotiw2015/Datasets/registered48x48'
target_file = '/home/usr/data/emotiw2015/Datasets/convnet_full_train_data.npz'

numpy_rng = np.random.RandomState(1)

fer_path = (os.path.join(data_main_folder, 'fer2013_images_histeq'))
tfd_path = (os.path.join(data_main_folder, 'TFD_reg'))
emotiw_train_path = (os.path.join(data_main_folder, 'Emotiw15_Train_reg'))
emotiw_val_path = (os.path.join(data_main_folder, 'Emotiw15_Val_reg'))


dataset = EmotionData(
    numpy_rng=numpy_rng, fer_path=fer_path, tfd_path=tfd_path,
    emotiw_train_path=emotiw_train_path, emotiw_val_path=emotiw_val_path,
    emotiw_test_path='',
    partitioning='convnet_full_train')

np.savez(target_file,
         train_inputs=dataset.train_inputs, train_outputs=dataset.train_outputs,
         val_inputs=dataset.val_inputs, val_outputs=dataset.val_outputs)

