import os
import shutil

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import tables
from theano.tensor.shared_randomstreams import RandomStreams

from disptools import tile_raster_images
from emoti_convnet_flip import EmotionConvNet
from trainer import SGD_Trainer

os.system('mkdir vis')

plt.gray()

numpy_rng = np.random.RandomState(1)
theano_rng = RandomStreams(1)

print 'loading data...'
full_data = np.load('/home/saebr/data/emotiw2015/Datasets/convnet_full_train_data.npz')
train_inputs = full_data['train_inputs']
train_outputs = full_data['train_outputs']
emoti_trainval_inputs = full_data['val_inputs']
emoti_trainval_outputs = full_data['val_outputs']

print 'normalizing data...'
with tables.openFile(
    '/home/saebr/data/emotiw2015/Datasets/fertfd_mean_std.h5'
) as h5file:
    mean0 = h5file.root.mean0.read()
    std0 = h5file.root.std0.read()

train_inputs = ((train_inputs - mean0) / std0).astype(np.float32)
emoti_trainval_inputs = ((emoti_trainval_inputs - mean0) / std0).astype(np.float32)

print 'instantiating model...'
model = EmotionConvNet(name='LastTest', numpy_rng=numpy_rng,
                       theano_rng=theano_rng,
                       batchsize=128)

print 'instantiating trainer...'
trainer = SGD_Trainer(
    model=model, inputs=train_inputs.reshape(
        train_inputs.shape[0], 1, 48, 48), outputs=train_outputs,
    batchsize=model.batchsize, learningrate=.001, momentum=.9,
    loadsize=train_inputs.shape[0], rng=numpy_rng, gradient_clip_threshold=np.inf,
    monitor_update_weight_norm_ratio=True
)
print 'computing 01 losses...'

# get best emoti_trainval_losses
best_emoti_trainval_loss = model.compute_01_loss(emoti_trainval_inputs, emoti_trainval_outputs)
print 'emoti_trainval 01 loss : {0}'.format(best_emoti_trainval_loss)

best_emoti_trainval_loss_epoch = 0

train_loss = model.compute_01_loss(train_inputs, train_outputs)

train_losses = []
emoti_trainval_losses = []
print 'train 01 loss: {0}'.format(train_loss)
for epoch in range(700):
    if epoch == 20:
        trainer.learningrate *= 1
    trainer.step()
    model.save('emoti_convnet.h5')
    plt.imsave('vis/convnet_filters.png', tile_raster_images(
        model.layers['conv0'].W.get_value(),
        model.layers['conv0'].filter_shape[2:], (8, 8), (1, 1)))

    print 'computing 01 losses...'
    emoti_trainval_loss = model.compute_01_loss(emoti_trainval_inputs, emoti_trainval_outputs)
    train_loss = model.compute_01_loss(train_inputs, train_outputs)

    train_losses.append(train_loss)
    emoti_trainval_losses.append(emoti_trainval_loss)

    # save if we found a new validation best with one of the two validation
    # sets
    if emoti_trainval_loss < best_emoti_trainval_loss:
        model.save('emoti_convnet_emoti_trainval_best.h5')
        best_emoti_trainval_loss = emoti_trainval_loss
        best_emoti_trainval_loss_epoch = epoch

    print 'emoti_trainval 01 loss: {0} (best so far: {1} at epoch {2})'.format(
        emoti_trainval_loss, best_emoti_trainval_loss, best_emoti_trainval_loss_epoch)
    print 'train 01 loss: {0}'.format(train_loss)


    try:
        shutil.copyfile('loss_log.h5', 'loss_log.h5_bak')
    except IOError:
        pass
    with tables.openFile('loss_log.h5', 'w') as logfile:
        logfile.createArray(logfile.root, 'train_losses', train_losses)
        logfile.createArray(logfile.root, 'emoti_trainval_losses', emoti_trainval_losses)

    plt.clf()
    plt.plot(np.arange(epoch + 1), train_losses, label='fertfd');
    plt.xlabel('epoch')
    plt.ylabel('01 loss')
    plt.plot(np.arange(epoch + 1), emoti_trainval_losses, label='EmotiW train+val')
    plt.legend()
    plt.savefig('vis/loss_plot.png')
