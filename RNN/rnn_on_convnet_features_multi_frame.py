import importlib
import os
import shutil
import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import tables
import theano
from theano.tensor.shared_randomstreams import RandomStreams

from trainer import SGD_Trainer
from rnn_multi_frame import RNN

os.system('mkdir -p vis')

numpy_rng = np.random.RandomState(1)
theano_rng = RandomStreams(1)
imshape = (48, 48)
nclass = 7
nhid = 200
window_size = 1
feature_layer_name = 'bias1'
data_file = '/data/usr/data/emotiw2015/Datasets/emotiw2015_rnn_early_stopping_data.npz'

feature_model_name = 'Exp5FliphCropDropShallowLargeFilterValAllEmoti'
feature_model_src = '../{0}/emoti_convnet_flip'.format(feature_model_name)
feature_model_params = '../{0}/emoti_convnet_emoti_trainval_best.h5'.format(feature_model_name)
feature_file_name = '/data/usr/data/emotiw2015/convfeatures/{0}_{1}.npz'.format(
    feature_model_name, feature_layer_name)
try:
    #always overwrite data
    #raise IOError()
    print 'trying to load data...'
    feature_data = np.load(feature_file_name)

    train_features = feature_data['train_features']
    val_features = feature_data['val_features']
    train_outputs = feature_data['train_outputs']
    val_outputs = feature_data['val_outputs']
    feature_size = train_features[0].shape[1:]
    print 'done'
except IOError:
    print 'couldn\'t load data, extracting...'
    sys.path.insert(0, os.path.split(feature_model_src)[0])
    print 'sys.path: {0}'.format(sys.path, )
    feature_model_module = importlib.import_module(os.path.split(feature_model_src)[1])
    feature_model = feature_model_module.EmotionConvNet(
        name='feature_model', numpy_rng=numpy_rng, theano_rng=theano_rng,
        batchsize=128)

    # for now set compute test value to warn, because we haven't specified test
    # values for the rnn
    theano.config.compute_test_value = 'ignore'

    print 'loading feature model params...'
    feature_model.load(feature_model_params)

    # get size of feature layer
    feature_layer = feature_model.layers[feature_layer_name]
    feature_size = feature_layer.outputs_shape[1:]
    print 'feature_size: {0}'.format(feature_size, )

    print 'compiling function for feature extraction...'
    feature_fn = theano.function(
        [feature_model.inputs], feature_layer.outputs)

    def get_features(inputs):
        batchsize = feature_model.batchsize

        # switch to test mode
        feature_model.mode.set_value(np.int8(1))

        print 'shape of inputs: {0}'.format(inputs.shape, )
        inputs = inputs.reshape((inputs.shape[0], ) +
                                feature_model.layers.values()[0].inputs_shape[1:])
        print 'shape of inputs: {0}'.format(inputs.shape, )
        features = np.empty(
            (len(inputs),) + feature_size, dtype=np.float32)
        print 'shape of features: {0}'.format(features.shape, )

        for b in range(inputs.shape[0] // batchsize):
            features[
                b * batchsize:
                (b + 1) * batchsize] = feature_fn(
                    inputs[b * batchsize: (b + 1) * batchsize])
        if inputs.shape[0] % batchsize:
            last_batch = np.zeros(feature_model.layers.values()[0].inputs_shape,
                                    dtype=theano.config.floatX)
            last_batch[:(inputs.shape[0] % batchsize)] = \
                inputs[-(inputs.shape[0] % batchsize):]
            features[-(inputs.shape[0] % batchsize):] = feature_fn(
                last_batch)[:(inputs.shape[0] % batchsize)]
        return features

    print 'loading data...'

    data = np.load(data_file)

    train_seq_names = data['train_seq_names']
    train_inputs = data['train_inputs']
    train_outputs = data['train_outputs']
    val_seq_names = data['val_seq_names']
    val_inputs = data['val_inputs']
    val_outputs = data['val_outputs']
    
    # get features

    print 'extracting features...'
    print '... for train...'
    train_features = []
    for s in range(len(train_inputs)):
        print '{0}/{1}'.format(s+1, len(train_inputs))
        train_features.append(get_features(train_inputs[s]))
    train_features = np.array(train_features)

    print '... for val...'
    val_features = []
    for s in range(len(val_inputs)):
        print '{0}/{1}'.format(s+1, len(val_inputs))
        val_features.append(get_features(val_inputs[s]))
    val_features = np.array(val_features)
    print 'done'
    np.savez(feature_file_name,
             train_features=train_features, val_features=val_features,
             train_outputs=train_outputs, val_outputs=val_outputs)
    
print 'padding data...'
# pad sequences to max len
train_maxlen = 0
for seq in train_features:
    if len(seq) > train_maxlen:
        train_maxlen = len(seq)

train_masks = np.zeros((len(train_features), train_maxlen - window_size + 1), dtype=np.int8)

train_inputs = np.zeros((len(train_features), train_maxlen,) + feature_size, dtype=np.float32)
for s, seq in enumerate(train_features):
    train_inputs[s, :len(seq)] = seq
    train_masks[s, len(seq)-window_size] = 1

train_outputs = train_outputs.reshape(-1, 1).repeat(train_maxlen - window_size + 1, axis=1)

val_maxlen = 0
for seq in val_features:
    if len(seq) > val_maxlen:
        val_maxlen = len(seq)

val_masks = np.zeros((len(val_features), val_maxlen - window_size + 1), dtype=np.int8)

val_inputs = np.zeros((len(val_features), val_maxlen,) + feature_size, dtype=np.float32)
for s, seq in enumerate(val_features):
    val_inputs[s, :len(seq)] = seq
    val_masks[s, len(seq)-window_size] = 1

val_outputs = val_outputs.reshape(-1, 1).repeat(val_maxlen - window_size + 1, axis=1)

train_inputs = train_inputs.reshape(
    train_inputs.shape[0], -1)

val_inputs = val_inputs.reshape(
    val_inputs.shape[0], -1)

print 'instantiating model...'
model = RNN(nin=np.prod(feature_size), nout=7, nhid=nhid, numpy_rng=numpy_rng, scale=1.0,
            window_size=window_size, theano_rng=theano_rng, dropout_rate=.0)
print 'done'

print 'instantiating trainer...'
trainer = SGD_Trainer(
    model=model, inputs=train_inputs,
    learningrate=.005, momentum=.0, batchsize=64, loadsize=711,
    gradient_clip_threshold=1., monitor_update_weight_norm_ratio=True,
    outputs=train_outputs, masks=train_masks)
print 'done'


train_losses = []
val_losses = []
best_val_01_loss = np.inf
best_val_01_loss_epoch = 0

for epoch in range(1000):
    trainer.step()
    model.save('rnn_params.h5')

    train_classifications = model.get_classifications(train_inputs)
    train_01_loss = np.sum(np.not_equal(train_classifications, train_outputs) *
                               train_masks) / np.float32(len(train_features))

    print 'train_01_loss: {0}'.format(train_01_loss, )
    val_classifications = model.get_classifications(val_inputs)
    val_01_loss = np.sum(np.not_equal(val_classifications, val_outputs) *
                               val_masks) / np.float32(len(val_features))

    train_losses.append(train_01_loss)
    val_losses.append(val_01_loss)
    if val_01_loss < best_val_01_loss:
        model.save('rnn_best_val_loss.h5')
        best_val_01_loss = val_01_loss
        best_val_01_loss_epoch = epoch


    print 'val_01 loss: {0} (best so far: {1} at epoch {2}, train 01 loss was {3})'.format(
        val_01_loss, best_val_01_loss, best_val_01_loss_epoch, train_losses[best_val_01_loss_epoch])

    try:
        shutil.copyfile('loss_log.h5', 'loss_log.h5_bak')
    except IOError:
        pass
    with tables.openFile('loss_log.h5', 'w') as logfile:
        logfile.createArray(logfile.root, 'train_losses', train_losses)
        logfile.createArray(logfile.root, 'val_losses', val_losses)

    plt.clf()
    plt.plot(np.arange(epoch+1), 1 - np.array(train_losses), label='EmotiW train')
    plt.plot(np.arange(epoch+1), 1 - np.array(val_losses), label='EmotiW val')
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('vis/lossplot_rnn.png')
