#!/usr/bin/env python
#-*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
import theano
theano.config.compute_test_value = 'raise'
import theano.tensor as T

from layers import (AffineLayer, ConvBiasLayer, ConvLayer, Dropout,
                    MaxPoolLayer, RandCropAndFlip, Relu, Softmax, Reshape)
from model import Model


class EmotionConvNet(Model):

    def __init__(self, name, numpy_rng, theano_rng, batchsize=128):
        # CALL PARENT CONSTRUCTOR TO SETUP CONVENIENCE FUNCTIONS
        # (SAVE/LOAD, ...)
        super(EmotionConvNet, self).__init__(name=name)

        self.numpy_rng = numpy_rng
        self.batchsize = batchsize
        self.theano_rng = theano_rng
        self.mode = theano.shared(np.int8(0), name='mode')

        self.inputs = T.ftensor4('inputs')
        self.inputs.tag.test_value = numpy_rng.randn(
            self.batchsize, 1, 48, 48).astype(np.float32)

        self.targets = T.ivector('targets')
        self.targets.tag.test_value = numpy_rng.randint(
            7, size=self.batchsize).astype(np.int32)

        self.layers = OrderedDict()
        
        self.layers['randcropandflip'] = RandCropAndFlip(
            inputs=self.inputs,
            image_shape=(self.batchsize, 1, 48, 48),
            patch_size=(44, 44),
            name='randcropandflip',
            theano_rng=self.theano_rng,
            mode_var=self.mode
        )
        
        self.layers['conv0'] = ConvLayer(
            rng=self.numpy_rng,
            inputs=self.layers['randcropandflip'],
            filter_shape=(32, 1, 9, 9),
            #image_shape=(self.batchsize, 1, 48, 48),
            name='conv0',
            pad=4
        )

        self.layers['maxpool0'] = MaxPoolLayer(
            inputs=self.layers['conv0'],
            pool_size=(2, 2),
            stride=(2, 2),
            name='maxpool0'
        )

        self.layers['bias0'] = ConvBiasLayer(
            inputs=self.layers['maxpool0'],
            name='bias0'
        )

        self.layers['relu0'] = Relu(
            inputs=self.layers['bias0'],
            name='relu0'
        )

        self.layers['dropout0'] = Dropout(
            inputs=self.layers['relu0'],
            dropout_rate=.25,
            name='dropout0',
            theano_rng=self.theano_rng,
            mode_var=self.mode
        )
        
        self.layers['conv1'] = ConvLayer(
            rng=self.numpy_rng,
            inputs=self.layers['dropout0'],
            filter_shape=(32, 32, 5, 5),
            name='conv1',
            pad=2
        )

        self.layers['maxpool1'] = MaxPoolLayer(
            inputs=self.layers['conv1'],
            pool_size=(2, 2),
            stride=(2, 2),
            name='maxpool1'
        )

        self.layers['bias1'] = ConvBiasLayer(
            inputs=self.layers['maxpool1'],
            name='bias1'
        )

        self.layers['relu1'] = Relu(
            inputs=self.layers['bias1'],
            name='relu1'
        )
        
        self.layers['dropout1'] = Dropout(
            inputs=self.layers['relu1'],
            dropout_rate=.25,
            name='dropout1',
            theano_rng=self.theano_rng,
            mode_var=self.mode
        )

        self.layers['conv2'] = ConvLayer(
            rng=self.numpy_rng,
            inputs=self.layers['dropout1'],
            filter_shape=(64, 32, 5, 5),
            name='conv2',
            pad=2
        )

        self.layers['maxpool2'] = MaxPoolLayer(
            inputs=self.layers['conv2'],
            pool_size=(2, 2),
            stride=(2, 2),
            name='maxpool2'
        )

        self.layers['bias2'] = ConvBiasLayer(
            inputs=self.layers['maxpool2'],
            name='bias2'
        )

        self.layers['relu2'] = Relu(
            inputs=self.layers['bias2'],
            name='relu2'
        )
        
        self.layers['dropout2'] = Dropout(
            inputs=self.layers['relu2'],
            dropout_rate=.25,
            name='dropout2',
            theano_rng=self.theano_rng,
            mode_var=self.mode
        )

        self.layers['reshape2'] = Reshape(
            inputs=self.layers['dropout2'],
            shape=(self.layers['dropout2'].outputs_shape[0],
                   np.prod(self.layers['dropout2'].outputs_shape[1:])),
            name='reshape2'
        )

        self.layers['fc3'] = AffineLayer(
            rng=self.numpy_rng,
            inputs=self.layers['reshape2'],
            nouts=7,
            name='fc3'
        )

        self.layers['softmax3'] = Softmax(
            inputs=self.layers['fc3'],
            name='softmax3'
        )

        self.probabilities = self.layers['softmax3'].outputs
        self.probabilities = T.clip(self.probabilities, 1e-6, 1-1e-6)

        self._cost = T.nnet.categorical_crossentropy(
            self.probabilities, self.targets).mean()

        self.classification = T.argmax(self.probabilities, axis=1)

        self.params = []
        for l in self.layers.values():
            self.params.extend(l.params)

        self._grads = T.grad(self._cost, self.params)

        self.classify = theano.function(
            [self.inputs], self.classification,
            #givens={self.mode: np.int8(1)})
            )

    def compute_01_loss(self, inputs, targets):
        # switch to test mode
        self.mode.set_value(np.int8(1))
        inputs = inputs.reshape((inputs.shape[0], ) +
                                self.layers.values()[0].inputs_shape[1:])
        predictions = np.zeros((inputs.shape[0],), dtype=np.int32)
        for b in range(inputs.shape[0] // self.batchsize):
            #print '{0}/{1}'.format(b+1, inputs.shape[0] // self.batchsize)
            predictions[
                b * self.batchsize:
                (b + 1) * self.batchsize] = self.classify(
                inputs[b * self.batchsize: (b + 1) * self.batchsize])
        if inputs.shape[0] % self.batchsize:
            last_batch = np.zeros(self.layers.values()[0].inputs_shape,
                                  dtype=theano.config.floatX)
            last_batch[:(inputs.shape[0] % self.batchsize)] = \
                inputs[-(inputs.shape[0] % self.batchsize):]
            predictions[-(inputs.shape[0] % self.batchsize):] = self.classify(
                last_batch)[:(inputs.shape[0] % self.batchsize)]
        # switch to train mode
        self.mode.set_value(np.int8(0))

        return np.sum(targets != predictions) / np.float32(inputs.shape[0])
