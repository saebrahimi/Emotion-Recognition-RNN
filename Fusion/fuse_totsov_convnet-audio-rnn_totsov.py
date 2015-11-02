import cPickle as pickle
from collections import OrderedDict
import os
import shutil
import sys
sys.path.append('..')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tables

from common.trainer import SGD_Trainer
from fusion import FusionNet


def compute_avg_probabilities(frame_probs):
    return np.array([x.mean(0) for x in frame_probs])

if __name__ == '__main__':
    
    feature_dicts = pickle.load(file('feature_dicts_totsov.pkl'))
    feature_dict_train = feature_dicts['feature_dict_train_totsov']
    feature_dict_val = feature_dicts['feature_dict_val_totsov']
    feature_dict_test = feature_dicts['feature_dict_test_totsov']
    modal_list = feature_dicts['modal_list']
    modal_list.remove('activity')
    
    # Fusion -----------------------------------------------------
    # Building prob vectors

    # probs_train contains one Nx7 matrix of probabilities for each modality
    probs_train = np.zeros((len(modal_list), len(feature_dict_train), 7), dtype=np.float32)
    target_train = np.empty(len(feature_dict_train), dtype=np.int32)
    for m, modal_name in enumerate(modal_list):
        for v, vid in enumerate(feature_dict_train.values()):
            probs_train[m, v] = vid['{0}_probs'.format(modal_name)]
            if m == 0:
                target_train[v] = vid['target']

    # build concatenated feature array
    modality_sizes = []
    features_train = []
    for v, vid in enumerate(feature_dict_train.values()):
        feature_vid = []
        for m, modal_name in enumerate(modal_list):
            feature_vid.append(vid['{0}_features'.format(modal_name)].reshape(1, -1))
            if v == 0:
                modality_sizes.append(feature_vid[-1].shape[1])
        features_train.append(np.hstack(feature_vid))
    train_inputs = np.vstack(features_train)

    probs_val = np.zeros((len(modal_list), len(feature_dict_val), 7), dtype=np.float32)
    target_val = np.empty(len(feature_dict_val), dtype=np.int32)
    for m, modal_name in enumerate(modal_list):
        for v, vid in enumerate(feature_dict_val.values()):
            probs_val[m, v] = vid['{0}_probs'.format(modal_name)]
            if m == 0:
                target_val[v] = vid['target']

    features_val = []
    for v, vid in enumerate(feature_dict_val.values()):
        feature_vid = []
        for m, modal_name in enumerate(modal_list):
            feature_vid.append(vid['{0}_features'.format(modal_name)].reshape(1, -1))
        features_val.append(np.hstack(feature_vid))
    val_inputs = np.vstack(features_val)
    
    probs_test = np.zeros((len(modal_list), len(feature_dict_test), 7), dtype=np.float32)
    for m, modal_name in enumerate(modal_list):
        for v, vid in enumerate(feature_dict_test.values()):
            probs_test[m, v] = vid['{0}_probs'.format(modal_name)]

    features_test = []
    for v, vid in enumerate(feature_dict_test.values()):
        feature_vid = []
        for m, modal_name in enumerate(modal_list):
            feature_vid.append(vid['{0}_features'.format(modal_name)].reshape(1, -1))
        features_test.append(np.hstack(feature_vid))
    test_inputs = np.vstack(features_test)

    print zip(modal_list, modality_sizes)
    
    os.system('mkdir -p vis')

    numpy_rng = np.random.RandomState(2)
        

    fusion_model = FusionNet(
        modality_names=modal_list,
        modality_sizes=modality_sizes,
        locallayer_sizes=((100,), (10,), (50,)),
        fusionlayer_sizes=(70,),
        numpy_rng=numpy_rng, batchsize=128)

    trainer = SGD_Trainer(
        model=fusion_model, inputs=train_inputs,
        learningrate=.1, momentum=.0,
        outputs=target_train, batchsize=fusion_model.batchsize,
        loadsize=len(feature_dict_train.keys()),
        rng=numpy_rng, gradient_clip_threshold=10.,
        monitor_update_weight_norm_ratio=True)

    best_emotiw_val_loss_epoch = 0
    best_emotiw_val_loss = np.inf

    train_losses = []
    val_losses = []

    for epoch in range(500):
        trainer.step()

        fusion_model.save('fuse_totsov_{0}_totsov.h5'.format('-'.join(modal_list)))
        val_loss = fusion_model.compute_01_loss(
            val_inputs, target_val)
        train_loss = fusion_model.compute_01_loss(
            train_inputs, target_train)
        val_losses.append(val_loss)
        train_losses.append(train_loss)

        if val_loss < best_emotiw_val_loss:
            fusion_model.save('fuse_totsov_{0}_totsov_val_best.h5'.format('-'.join(modal_list)))
            best_emotiw_val_loss = val_loss
            best_emotiw_val_loss_epoch = epoch

        print 'emotiw_val 01 loss: {0} (best so far: {1} at epoch {2}, emotiw_train 01 loss was {3})'.format(
            val_loss, best_emotiw_val_loss, best_emotiw_val_loss_epoch,
            train_losses[best_emotiw_val_loss_epoch])

        print 'emotiw_train 01 loss: {0}'.format(train_loss)

        try:
            shutil.copyfile('loss_log_totsov_{0}_totsov.h5'.format(
                '-'.join(modal_list)), 'loss_log_totsov_{0}_totsov.h5_bak'.format(
                '-'.join(modal_list)))
        except IOError:
            pass
        with tables.openFile('loss_log_totsov_{0}_totsov.h5'.format('-'.join(modal_list)), 'w') as logfile:
            logfile.createArray(logfile.root, 'train_losses', train_losses)
            logfile.createArray(logfile.root, 'val_losses', val_losses)

        plt.clf()
        plt.plot(np.arange(epoch + 1), train_losses, label='train');
        plt.xlabel('epoch')
        plt.ylabel('01 loss')
        plt.plot(np.arange(epoch + 1), val_losses, label='val')
        plt.legend()
        plt.savefig('vis/loss_totsov_plot_{0}_totsov.png'.format(
            '-'.join(modal_list)))
        
    # dump features
    print 'Extracting probs...'
    fusion_model.load('fuse_totsov_{0}_totsov_val_best.h5'.format(
        '-'.join(modal_list)))
    fuse_train_probs = fusion_model.get_probabilities(train_inputs)
    fuse_val_probs = fusion_model.get_probabilities(val_inputs)
    fuse_test_probs = fusion_model.get_probabilities(test_inputs)
    np.savez('fuse_totsov_{0}_totsov_probs_rectifiedtanh_dropout.npz'.format(
        '-'.join(modal_list)),
             train_probs=fuse_train_probs, train_targets=target_train,
             train_vid_ids=feature_dict_train.keys(),
             val_probs=fuse_val_probs, val_targets=target_val,
             val_vid_ids=feature_dict_val.keys(),
             test_probs=fuse_test_probs,
             test_vid_ids=feature_dict_test.keys())
    fusion_model.save('fuse_totsov_{0}_totsov_val_best_rectifiedtanh_dropout.h5'.format('-'.join(modal_list)))
