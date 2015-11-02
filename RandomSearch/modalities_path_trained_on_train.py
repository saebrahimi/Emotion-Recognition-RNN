import os

import numpy as np
from collections import OrderedDict

# load data from modalities Trained on Training EmotiW

# audio
# convnet (svm)
# rnn
# activity

modal_list = []

feature_dict_train = OrderedDict()
feature_dict_val = OrderedDict()
feature_dict_test = OrderedDict()

# load convnet_svm ---------------------------------------------------------

# we take convnet vid ids as reference
convnet_feature_data = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/convnet_svm/'
    'Exp5convnet_layer_fc3_fixedlen_svm_probs_and_features.npz')

# train
convnet_vid_ids_train = convnet_feature_data['train_vidids']
convnet_features_train = convnet_feature_data['train_features']
convnet_probs_train = convnet_feature_data['train_probs']
train_targets = convnet_feature_data['train_outputs']
convnet_feature_size = convnet_features_train.shape[1]

for train_id, train_feature, train_probs, train_target in zip(
    convnet_vid_ids_train, convnet_features_train, convnet_probs_train,
    train_targets):

    feature_dict_train[train_id] = OrderedDict()
    feature_dict_train[train_id]['convnet_features'] = train_feature
    feature_dict_train[train_id]['convnet_probs'] = train_probs
    feature_dict_train[train_id]['target'] = train_target

# val
convnet_vid_ids_val = convnet_feature_data['val_vidids']
convnet_features_val = convnet_feature_data['val_features']
convnet_probs_val = convnet_feature_data['val_probs']
val_targets = convnet_feature_data['val_outputs']

for val_id, val_feature, val_probs, val_target in zip(
    convnet_vid_ids_val, convnet_features_val, convnet_probs_val,
    val_targets):

    feature_dict_val[val_id] = OrderedDict()
    feature_dict_val[val_id]['convnet_features'] = val_feature
    feature_dict_val[val_id]['convnet_probs'] = val_probs
    feature_dict_val[val_id]['target'] = val_target
    
# test
convnet_vid_ids_test = convnet_feature_data['test_vidids']
convnet_features_test = convnet_feature_data['test_features']
convnet_probs_test = convnet_feature_data['test_probs']

for test_id, test_feature, test_probs in zip(
    convnet_vid_ids_test, convnet_features_test, convnet_probs_test):

    feature_dict_test[test_id] = OrderedDict()
    feature_dict_test[test_id]['convnet_features'] = test_feature
    feature_dict_test[test_id]['convnet_probs'] = test_probs

modal_list.append('convnet')

# load audio -----------------------------------------------------------

# train
audio_features_train = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'audio_bugfixed/pca_reduced10_train_data.npy')
audio_probs_train = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'audio_bugfixed/training_Tr_testing_V_train_probs.npy')
audio_filelist_train = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'audio_bugfixed/training_Tr_testing_V_train_file_lst.npy')
audio_labels_train = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'audio_bugfixed/training_Tr_testing_V_train_labels.npy')
audio_vid_ids_train = [os.path.splitext(os.path.split(
    vid)[1])[0] for vid in audio_filelist_train]
audio_feature_size = audio_features_train.shape[1]

for train_id, train_feature, train_probs, train_target in zip(
    audio_vid_ids_train, audio_features_train,
    audio_probs_train, audio_labels_train):
    if not train_id in feature_dict_train.keys():
        print '{0} not in feature_dict_train'.format(train_id)
        continue
    assert train_target==feature_dict_train[train_id]['target'],\
    'audio: {0},{1},{2}'.format(
        train_target, feature_dict_train[train_id]['target'], train_id)
    feature_dict_train[train_id]['audio_features'] = train_feature
    feature_dict_train[train_id]['audio_probs'] = train_probs

# val
audio_features_val = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'audio_bugfixed/pca_reduced10_valid_data.npy')
audio_probs_val = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'audio_bugfixed/training_Tr_testing_V_test_probs.npy')
audio_filelist_val = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'audio_bugfixed/training_Tr_testing_V_test_file_lst.npy')
audio_labels_val = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'audio_bugfixed/training_Tr_testing_V_test_labels.npy')
audio_vid_ids_val = [os.path.splitext(os.path.split(
    vid)[1])[0] for vid in audio_filelist_val]

for val_id, val_feature, val_probs, val_target in zip(
    audio_vid_ids_val, audio_features_val,
    audio_probs_val, audio_labels_val):
    if not val_id in feature_dict_val.keys():
        print '{0} not in feature_dict_val'.format(val_id)
        continue
    assert val_target==feature_dict_val[val_id]['target'],\
    'audio: {0},{1},{2}'.format(
        val_target, feature_dict_val[val_id]['target'], val_id)
    feature_dict_val[val_id]['audio_features'] = val_feature
    feature_dict_val[val_id]['audio_probs'] = val_probs
    
# test
audio_features_test = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'audio_bugfixed/pca_reduced10_test_data.npy')
audio_probs_test = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'audio_bugfixed/training_Tr_testing_Ts_test_probs.npy')
audio_filelist_test = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'audio_bugfixed/training_Tr_testing_Ts_test_file_lst.npy')
audio_vid_ids_test = [os.path.splitext(os.path.split(
    vid)[1])[0] for vid in audio_filelist_test]

for test_id, test_feature, test_probs in zip(
    audio_vid_ids_test, audio_features_test,
    audio_probs_test):
    if not test_id in feature_dict_test.keys():
        print '{0} not in feature_dict_test'.format(test_id)
        continue
    feature_dict_test[test_id]['audio_features'] = test_feature
    feature_dict_test[test_id]['audio_probs'] = test_probs

modal_list.append('audio')

# load rnn ---------------------------------------------------------

rnn_feature_data = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/rnn/'
    'rnn_hiddens_and_probs_Exp5FliphCropDropShallowLargeFilterValAllEmoti_bias1.npz')

# train
rnn_vid_ids_train = rnn_feature_data['train_seq_names']
rnn_features_train = rnn_feature_data['train_hiddens']
rnn_probs_train = rnn_feature_data['train_probs']
rnn_targets_train = rnn_feature_data['train_targets']
rnn_feature_size = rnn_features_train.shape[1]

for train_id, train_feature, train_probs, train_target in zip(
    rnn_vid_ids_train, rnn_features_train,
    rnn_probs_train, rnn_targets_train):
    if not train_id in feature_dict_train.keys():
        print '{0} not in feature_dict_train'.format(train_id)
        continue
    assert train_target==feature_dict_train[train_id]['target'],\
    'rnn: {0},{1},{2}'.format(
        train_target, feature_dict_train[train_id]['target'], train_id)
    feature_dict_train[train_id]['rnn_features'] = train_feature
    feature_dict_train[train_id]['rnn_probs'] = train_probs
    
# val
rnn_vid_ids_val = rnn_feature_data['val_seq_names']
rnn_features_val = rnn_feature_data['val_hiddens']
rnn_probs_val = rnn_feature_data['val_probs']
rnn_targets_val = rnn_feature_data['val_targets']

for val_id, val_feature, val_probs, val_target in zip(
    rnn_vid_ids_val, rnn_features_val,
    rnn_probs_val, rnn_targets_val):
    if not val_id in feature_dict_val.keys():
        print '{0} not in feature_dict_val'.format(val_id)
        continue
    assert val_target==feature_dict_val[val_id]['target'],\
    'rnn: {0},{1},{2}'.format(
        val_target, feature_dict_val[val_id]['target'], val_id)
    feature_dict_val[val_id]['rnn_features'] = val_feature
    feature_dict_val[val_id]['rnn_probs'] = val_probs
    
# test
rnn_vid_ids_test = rnn_feature_data['test_seq_names']
rnn_features_test = rnn_feature_data['test_hiddens']
rnn_probs_test = rnn_feature_data['test_probs']

for test_id, test_feature, test_probs in zip(
    rnn_vid_ids_test, rnn_features_test,
    rnn_probs_test):
    if not test_id in feature_dict_test.keys():
        print '{0} not in feature_dict_test'.format(test_id)
        continue
    feature_dict_test[test_id]['rnn_features'] = test_feature
    feature_dict_test[test_id]['rnn_probs'] = test_probs

modal_list.append('rnn')

# load activity -----------------------------------------------------------

# train
activity_features_train = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'activity/training_Tr_testing_V_train_data_kernel.npy')
activity_probs_train = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'activity/training_Tr_testing_V_train_probs.npy')
activity_filelist_train = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'activity/training_Tr_testing_V_train_file_lst.npy')
activity_labels_train = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'activity/training_Tr_testing_V_train_labels.npy')
activity_vid_ids_train = [os.path.split(
    vid)[1] for vid in activity_filelist_train]
activity_feature_size = activity_features_train.shape[1]

for train_id, train_feature, train_probs, train_target in zip(
    activity_vid_ids_train, activity_features_train,
    activity_probs_train, activity_labels_train):
    if not train_id in feature_dict_train.keys():
        print '{0} not in feature_dict_train'.format(train_id)
        continue
    assert train_target==feature_dict_train[train_id]['target'],\
    'activity: {0},{1},{2}'.format(
        train_target, feature_dict_train[train_id]['target'], train_id)
    feature_dict_train[train_id]['activity_features'] = train_feature
    feature_dict_train[train_id]['activity_probs'] = train_probs

# val
activity_features_val = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'activity/training_Tr_testing_V_test_data_kernel.npy')
activity_probs_val = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'activity/training_Tr_testing_V_test_probs.npy')
activity_filelist_val = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'activity/training_Tr_testing_V_test_file_lst.npy')
activity_labels_val = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'activity/training_Tr_testing_V_test_labels.npy')
activity_vid_ids_val = [os.path.split(
    vid)[1] for vid in activity_filelist_val]

for val_id, val_feature, val_probs, val_target in zip(
    activity_vid_ids_val, activity_features_val,
    activity_probs_val, activity_labels_val):
    if not val_id in feature_dict_val.keys():
        print '{0} not in feature_dict_val'.format(val_id)
        continue
    assert val_target==feature_dict_val[val_id]['target'],\
    'activity: {0},{1},{2}'.format(
        val_target, feature_dict_val[val_id]['target'], val_id)
    feature_dict_val[val_id]['activity_features'] = val_feature
    feature_dict_val[val_id]['activity_probs'] = val_probs
    
# test
activity_features_test = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'activity/training_Tr_testing_Ts_test_data_kernel.npy')
activity_probs_test = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'activity/training_Tr_testing_Ts_test_probs.npy')
activity_filelist_test = np.load(
    '/home/usr/data/emotiw2015/fusion/trained_on_train_emotiw/'
    'activity/training_Tr_testing_Ts_test_file_lst.npy')
activity_vid_ids_test = [os.path.split(
    vid)[1] for vid in activity_filelist_test]

for test_id, test_feature, test_probs in zip(
    activity_vid_ids_test, activity_features_test,
    activity_probs_test):
    if not test_id in feature_dict_test.keys():
        print '{0} not in feature_dict_test'.format(test_id)
        continue
    feature_dict_test[test_id]['activity_features'] = test_feature
    feature_dict_test[test_id]['activity_probs'] = test_probs

modal_list.append('activity')
