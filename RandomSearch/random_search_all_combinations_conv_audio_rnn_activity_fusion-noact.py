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
from random_search import random_search, random_search_gpu, local_random_search
from modalities_path_trained_on_train import *

def compute_avg_probabilities(frame_probs):
    return np.array([x.mean(0) for x in frame_probs])

if __name__ == '__main__':

    # load fusion all except GC and activity -------------------------------

    fusion_feature_data = np.load(
        '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/fusion/fuse_convnet_audio_rnn_probs.npz')

    # train
    fusion_vid_ids_train = fusion_feature_data['train_vid_ids']
    fusion_probs_train = fusion_feature_data['train_probs']
    fusion_targets_train = fusion_feature_data['train_targets']

    for train_id, train_probs, train_target in zip(
        fusion_vid_ids_train, fusion_probs_train, fusion_targets_train):
        if not train_id in feature_dict_train.keys():
            print '{0} not in feature_dict_train'.format(train_id)
            continue
        assert train_target==feature_dict_train[train_id]['target'],\
        'fusion: {0},{1},{2}'.format(
            train_target, feature_dict_train[train_id]['target'], train_id)
        feature_dict_train[train_id]['fusion_probs'] = train_probs
        
    # val
    fusion_vid_ids_val = fusion_feature_data['val_vid_ids']
    fusion_probs_val = fusion_feature_data['val_probs']
    fusion_targets_val = fusion_feature_data['val_targets']
 
    for val_id, val_probs, val_target in zip(
        fusion_vid_ids_val, fusion_probs_val, fusion_targets_val):
        if not val_id in feature_dict_val.keys():
            print '{0} not in feature_dict_val'.format(val_id)
            continue
        assert val_target==feature_dict_val[val_id]['target'],\
        'fusion: {0},{1},{2}'.format(
            val_target, feature_dict_val[val_id]['target'], val_id)
        feature_dict_val[val_id]['fusion_probs'] = val_probs
        
    # test
    fusion_vid_ids_test = fusion_feature_data['test_vid_ids']
    fusion_probs_test = fusion_feature_data['test_probs']
    
    for test_id, test_probs in zip(
        fusion_vid_ids_test, fusion_probs_test):
        if not test_id in feature_dict_test.keys():
            print '{0} not in feature_dict_test'.format(test_id)
            continue
        feature_dict_test[test_id]['fusion_probs'] = test_probs
    
    modal_list.append('fusion')

    n_modalities = len(modal_list)
    selection_masks = np.unpackbits(np.arange(
        2**n_modalities, dtype=np.uint8).reshape(2**n_modalities,1), 
        axis=1)[:,8-n_modalities:]

    for mask_idx, mask in enumerate(selection_masks):
        if mask.sum() < 2:
            continue

        modal_list_masked = np.array(modal_list)[np.where(mask == 1)]

        # Random Search-----------------------------------------------------
        # Building prob vectors

        # probs_train contains one Nx7 matrix of probabilities for each modality
        probs_train = np.zeros((len(modal_list_masked), len(feature_dict_train), 7), dtype=np.float32)
        target_train = np.empty(len(feature_dict_train), dtype=np.int32)
        for m, modal_name in enumerate(modal_list_masked):
            for v, vid in enumerate(feature_dict_train.values()):
                probs_train[m, v] = vid['{0}_probs'.format(modal_name)]
                if m == 0:
                    target_train[v] = vid['target']

        probs_val = np.zeros((len(modal_list_masked), len(feature_dict_val), 7), dtype=np.float32)
        target_val = np.empty(len(feature_dict_val), dtype=np.int32)
        for m, modal_name in enumerate(modal_list_masked):
            for v, vid in enumerate(feature_dict_val.values()):
                probs_val[m, v] = vid['{0}_probs'.format(modal_name)]
                if m == 0:
                    target_val[v] = vid['target']
                    
        probs_test = np.zeros((len(modal_list_masked), len(feature_dict_test), 7), dtype=np.float32)
        for m, modal_name in enumerate(modal_list_masked):
            for v, vid in enumerate(feature_dict_test.values()):
                probs_test[m, v] = vid['{0}_probs'.format(modal_name)]
     
        os.system('mkdir -p vis weights')

        numpy_rng = np.random.RandomState(1)
        best_weights, best_iter, best_acc = random_search(
            probs_train, probs_val,
            target_train, target_val,
            numpy_rng, n_iter=100000)
        print best_weights, best_iter, best_acc
        std = .5
        while std > 1e-4:
            print 'std {0}'.format(std)
            best_weights, best_iter, best_acc = local_random_search(
                probs_train, probs_val, probs_test,
                feature_dict_test.keys(),
                target_train, target_val,
                numpy_rng, n_iter=100000,
                mu=best_weights, std=std, best_acc=best_acc,
                modalities='_'.join(modal_list_masked))
            std *= .9
            print best_weights, best_iter, best_acc
        

