#!/usr/bin/env python
#-*- coding: utf-8 -*-

from glob import glob
import os
from PIL import Image

import numpy as np


class EmotionData(object):

    labels = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

    imshape = (48, 48)


    def __init__(self, numpy_rng, fer_path, tfd_path, emotiw_train_path,
                 emotiw_val_path, emotiw_test_path, partitioning, imshape=None):
        """
        Args
        ----
            numpy_rng: numpy random number generator
            fer_path: path to fer data
            tfd_path: path to tfd data
            emotiw_train_path: path to emotiw_train data
            emotiw_val_path: path to emotiw_val data
            emotiw_test_path: path to emotiw_test data
            partitioning: which split to use
                'convnet' is used to train the convnet
                'convnet_full_train'
                'rnn_early_stopping' is used to find the epoch in which to
                    stop training
                'emotiw_train_val_test' is used after we found the early stopping
                    epoch to train with all allowed training data
        """
        assert partitioning in ('convnet', 
                                'convnet_full_train',
                                'rnn_early_stopping',
                                'emotiw_train_val_test')

        self.numpy_rng = numpy_rng
        self.fer_path = fer_path
        self.tfd_path = tfd_path
        self.emotiw_train_path = emotiw_train_path
        self.emotiw_val_path = emotiw_val_path
        self.emotiw_test_path = emotiw_test_path
        self.partitioning = partitioning
        if imshape is not None:
            self.imshape = imshape

        print 'loading data for {0} split...'.format(partitioning)
        if partitioning == 'convnet':
            (self.train_inputs,
             self.train_outputs) = self.load_set(
                 (self.fer_path, self.tfd_path))
            (self.val_inputs,
             self.val_outputs) = self.load_set(
                 (self.emotiw_train_path, ))
        elif partitioning == 'convnet_full_train':
            (self.train_inputs,
             self.train_outputs) = self.load_set(
                 (self.fer_path, self.tfd_path))
            (self.val_inputs,
             self.val_outputs) = self.load_set(
                 (self.emotiw_train_path, self.emotiw_val_path))
        elif partitioning == 'rnn_early_stopping':
            (self.train_seq_names,
             self.train_inputs,
             self.train_outputs) = self.load_sequence_set(
                 (self.emotiw_train_path, ))
            (self.val_seq_names,
             self.val_inputs,
             self.val_outputs) = self.load_sequence_set(
                 (self.emotiw_val_path, ))
        elif partitioning == 'emotiw_train_val_test':
            (self.train_seq_names,
             self.train_inputs,
             self.train_outputs) = self.load_sequence_set(
                 (self.emotiw_train_path, ))
            (self.val_seq_names,
             self.val_inputs,
             self.val_outputs) = self.load_sequence_set(
                 (self.emotiw_val_path, ))
            (self.test_seq_names,
             self.test_inputs) = self.load_sequence_set(
                 (self.emotiw_test_path, ))
            
        # shuffle training and validation set
        print 'shuffling training set...'
        shuffle_indices = numpy_rng.permutation(self.train_inputs.shape[0])
        self.train_inputs = self.train_inputs[shuffle_indices]
        self.train_outputs = self.train_outputs[shuffle_indices]
        self.train_seq_names = self.train_seq_names[shuffle_indices]
        print 'done'

        print 'shuffling validation set...'
        shuffle_indices = numpy_rng.permutation(self.val_inputs.shape[0])
        self.val_inputs = self.val_inputs[shuffle_indices]
        self.val_outputs = self.val_outputs[shuffle_indices]
        self.val_seq_names = self.val_seq_names[shuffle_indices]
        print 'done'


    def load_set(self, path_list):
        inputs = []
        outputs = []
        for set_path in path_list:
            inp, outp = self.load_from_path(set_path)
            inputs.append(inp)
            outputs.append(outp)
        return (np.concatenate(inputs, axis=0),
                np.concatenate(outputs, axis=0))


    def load_sequence_set(self, path_list):
        seq_names = []
        inputs = []
        is_unlabeled = False

        outputs = []
        for set_path in path_list:
            print set_path
            res = self.load_sequences_from_path(set_path)
            if len(res) == 3:
                set_seq_names, inp, outp = res
                outputs.append(outp)
            else:
                is_unlabeled = True
                set_seq_names, inp = res
            seq_names.append(set_seq_names)
            inputs.append(inp)
        if is_unlabeled:
            return (np.concatenate(seq_names, axis=0),
                    np.concatenate(inputs, axis=0))
        else:
            return (np.concatenate(seq_names, axis=0),
                    np.concatenate(inputs, axis=0),
                    np.concatenate(outputs, axis=0))


    def load_sequences_from_path(self, path):
        inputs = []
        outputs = []
        files = sorted(
            map(lambda x: os.path.split(x)[1],
                glob(os.path.join(path, '*'))))
        seq_names = np.unique(map(lambda x: x.split('_', 1)[0], files))

        is_unlabeled = False
        for s, seq in enumerate(seq_names):
            print 'processing sequence {0}/{1}'.format(s+1, len(seq_names))

            seq_frames = filter(lambda x: x.startswith(seq), files)
            
            label_tmp = os.path.splitext(seq_frames[0])[0].split('_')[-1]
            if label_tmp == '':
                is_unlabeled = True
            else:
                label = int(label_tmp)
            
            #print s, seq, seq_frames, label

            seq_inputs = np.empty(
                (len(seq_frames), ) + self.imshape,
                dtype=np.uint8)
            for f, frame in enumerate(seq_frames):
                seq_inputs[f] = np.array(
                    Image.open(os.path.join(path, frame)).convert('L'))
            inputs.append(seq_inputs)
            if not is_unlabeled:
                outputs.append(label)

        if is_unlabeled:
            return np.array(seq_names), np.array(inputs)
        else:
            return np.array(seq_names), np.array(inputs), np.array(
                outputs, dtype=np.int32)


    def load_from_path(self, path):
        # get list of files and allocate space for inputs/outputs
        files = glob(os.path.join(path, '*.png'))
        inputs = np.empty(
            (len(files), ) + self.imshape,
            dtype=np.uint8)
        outputs = np.empty(
            (len(files), ), dtype=np.int32)

        print 'loading data from {0}...'.format(path)

        for i, fname in enumerate(files):
            if (i + 1) % 100 == 0:
                print '{0}/{1}'.format(i+1, len(files))
            inputs[i] = np.array(Image.open(fname).convert('L'))
            outputs[i] = int(os.path.splitext(fname)[0][-1])
        return inputs, outputs


if __name__ == '__main__':

    numpy_rng = np.random.RandomState(1)

    #data_main_folder = '/home/vincent/data/emotiw2015/Datasets/registered48x48'
    data_main_folder = '/data/lisatmp3/michals/data/emotiw2015/Datasets/registered48x48'

    # NOTE: THIS IS THE PARTITIONING FOR CONVNET TRAINING,
    #       FOR RNN TRAINING WE
    fer_path = (os.path.join(data_main_folder, 'fer2013_images_histeq'))
    tfd_path = (os.path.join(data_main_folder, 'TFD_reg'))
    emotiw_train_path = (os.path.join(data_main_folder, 'Emotiw15_Train_reg'))
    emotiw_val_path = (os.path.join(data_main_folder, 'Emotiw15_Val_reg'))
    emotiw_test_path = (os.path.join(data_main_folder, 'Emotiw15_Test_reg'))

    dataset = EmotionData(
        numpy_rng=numpy_rng, fer_path=fer_path, tfd_path=tfd_path,
        emotiw_train_path=emotiw_train_path, emotiw_val_path=emotiw_val_path,
        emotiw_test_path=emotiw_test_path,
        partitioning='rnn_early_stopping')

# vim: set ts=4 sw=4 sts=4 expandtab:
