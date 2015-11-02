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
                 emotiw_val_path, partitioning):
        """
        Args
        ----
            numpy_rng: numpy random number generator
            fer_path: path to fer data
            tfd_path: path to tfd data
            emotiw_train_path: path to emotiw_train data
            emotiw_val_path: path to emotiw_val data
            partitioning: which split to use
                'convnet' is used to train the convnet
                'rnn_early_stopping' is used to find the epoch in which to
                    stop training
                'rnn_full_train' is used after we found the early stopping
                    epoch to train with all allowed training data
        """
        assert partitioning in ('convnet', 'rnn_early_stopping',
                                'rnn_full_train')

        self.numpy_rng = numpy_rng
        self.fer_path = fer_path
        self.tfd_path = tfd_path
        self.emotiw_train_path = emotiw_train_path
        self.emotiw_val_path = emotiw_val_path

        print 'loading data for {0} split...'.format(partitioning)
        if partitioning == 'convnet':
            (self.train_inputs,
             self.train_outputs) = self.load_set(
                 (self.fer_path, self.tfd_path))
            (self.val_inputs,
             self.val_outputs) = self.load_set(
                 (self.emotiw_train_path, ))
        elif partitioning == 'rnn_early_stopping':
            # TODO: RNN partitioning
            raise NotImplementedError()
            (self.train_inputs,
             self.train_outputs) = self.load_sequence_set(
                 (self.emotiw_train_path, ))
            (self.val_inputs,
             self.val_outputs) = self.load_sequence_set(
                 (self.emotiw_val_path, ))
        elif partitioning == 'rnn_full_train':
            raise NotImplementedError()
            (self.train_inputs,
             self.train_outputs) = self.load_sequence_set(
                 (self.emotiw_train_path, self.emotiw_val_path))

        # shuffle training set
        print 'shuffling training set...'
        shuffle_indices = numpy_rng.permutation(self.train_inputs.shape[0])
        self.train_inputs = self.train_inputs[shuffle_indices]
        self.train_outputs = self.train_outputs[shuffle_indices]
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
        # TODO: should load sequence data sets via load_sequences_from_path()
        #       collect in list and DO NOT STACK!!!
        #       sequences are padded with zeros only when we build a batch!
        raise NotImplementedError()


    def load_sequences_from_path(self, path):
        raise NotImplementedError()


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

# vim: set ts=4 sw=4 sts=4 expandtab:
