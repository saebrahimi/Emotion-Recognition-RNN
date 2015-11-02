#!/usr/bin/env python
#-*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from common.layers import *
from common.model import Model
from common.objectives import Frobenius, L11, L2_sqr, L21


class FusionNet(Model):
    """Fusion network as described in
    Zuxuan Wu, Xi Wang, Yu-Gang Jiang, Hao Ye, Xiangyang Xue -
    'Modeling Spatial-Temporal Clues in a Hybrid Deep Learning
    Framework for Video Classification'
    """
    def __init__(self, modality_names, modality_sizes,
                 locallayer_sizes, fusionlayer_sizes,
                 numpy_rng, batchsize, theano_rng=None):
        assert (len(modality_names)==len(modality_sizes) and
                len(modality_names)==len(locallayer_sizes))
        self.modality_names = modality_names
        self.modality_sizes = modality_sizes
        self.locallayer_sizes = locallayer_sizes
        self.fusionlayer_sizes = fusionlayer_sizes
        self.numpy_rng = numpy_rng
        self.batchsize = batchsize
        if theano_rng is None:
            theano_rng = RandomStreams(1)
        self.theano_rng = theano_rng
        self.mode = theano.shared(np.int8(0), name='mode')

        # start with empty params list
        self.params = []
        self.l1params = []
        self.l2params = []
        self.l21params = []
        
        # inputs are the concatenated modalities
        self.inputs = T.fmatrix('inputs')

        # targets vector
        self.targets = T.ivector('targets')

        self.modality_inputs = OrderedDict()
        self.modality_models = OrderedDict()
        self.modality_preconcat_layer_sizes = []
        self.modality_concat_layer_sizes = []

        offset = 0
        # local modality networks
        for modality_name, modality_size, locallayer_size in zip(
            modality_names, modality_sizes, locallayer_sizes):

            # get inputs of modality
            self.modality_inputs[modality_name] = self.inputs[:, offset:offset+modality_size]
            offset += modality_size
            
            # determine size of input to the last layer in the modalities subnetwork
            if len(locallayer_size) == 1:
                self.modality_preconcat_layer_sizes.append(modality_size)
            else:
                self.modality_preconcat_layer_sizes.append(locallayer_size[-2])

            # construct modality model
            layers = []
            #locallayer_sizes = ((100,), (100,200)) 
            #locallayer_size = (100,)
            for i, size in enumerate(locallayer_size):
                if i == 0:
                    layer_input = self.modality_inputs[modality_name]
                    layer_input_size = (self.batchsize, modality_size)
                else:
                    layer_input = layers[-1]
                    layer_input_size = layer_input.outputs_shape
                layers.append(AffineLayer(
                    rng=self.numpy_rng, 
                    inputs=layer_input,
                    nouts=size,
                    name='{0}_affine_{1}'.format(modality_name, i),
                    inputs_shape=layer_input_size))
                # append params to global list
                self.params.extend(layers[-1].params)
                self.l2params.append(layers[-1].W)
                if i==len(locallayer_size)-1:
                    self.l1params.append(layers[-1].W)
                    self.l21params.append(layers[-1].W)
                    # update total size of concat layer
                    self.modality_concat_layer_sizes.append(size)
                layers.append(RectifiedTanh(
                    inputs=layers[-1],
                    name='{0}_rectifiedtanh_{1}'.format(modality_name, i)))
            # create the modality model object
            self.modality_models[modality_name] = Composite(
                layers=layers, name='{0}_composite'.format(modality_name))
        # concatenate modality model outputs
        self.concat_modalities = Concat(
            self.modality_models.values(), name='concat_layer', axis=1)
        self.fusion_layers = []
        for i, fusionlayer_size in enumerate(fusionlayer_sizes):
            if i == 0:
                layer_input = self.concat_modalities
            else:
                layer_input = self.fusion_layers[-1]
            self.fusion_layers.append(AffineLayer(
                rng=self.numpy_rng, 
                inputs=layer_input,
                nouts=fusionlayer_size,
                name='fusion_affine_{0}'.format(i)))
            # append params to global list
            self.params.extend(self.fusion_layers[-1].params)
            self.l2params.append(self.fusion_layers[-1].W)
            self.fusion_layers.append(RectifiedTanh(
                inputs=self.fusion_layers[-1],
                name='fusion_rectifiedtanh_{0}'.format(i)))

            self.fusion_layers.append(Dropout(
                inputs=self.fusion_layers[-1],
                dropout_rate=.3,
                name='fusion_dropout_{0}'.format(i),
                theano_rng=self.theano_rng,
                mode_var=self.mode))
        # classification layer
        self.logits = AffineLayer(
            rng=self.numpy_rng,
            inputs=self.fusion_layers[-1],
            nouts=7,
            name='logit_affine')
        # append params to global list
        self.params.extend(self.logits.params)
        self.l2params.append(self.logits.W)
        self.softmax = Softmax(
            inputs=self.logits,
            name='softmax')

        self.probabilities = self.softmax.outputs
        self.probabilities = T.clip(self.probabilities, 1e-6, 1-1e-6)
        
        self.l2cost = L2_sqr(T.concatenate(
            [x.flatten() for x in self.l2params], axis=0))
        
        self.concat_matrix = T.zeros(
            (np.sum(self.modality_preconcat_layer_sizes),
             np.sum(self.modality_concat_layer_sizes)))
        row_offset = 0
        col_offset = 0
        for inp_size, outp_size, p in zip(self.modality_preconcat_layer_sizes,
                                          self.modality_concat_layer_sizes,
                                          self.l1params):
            # embed weight matrices in large concatenated matrix
            self.concat_matrix = T.set_subtensor(
                self.concat_matrix[row_offset:row_offset+inp_size,
                                   col_offset:col_offset+outp_size], p)
        self.l1cost = L11(self.concat_matrix)
        self.l21cost = L21(self.concat_matrix)

        self._cost = (T.nnet.categorical_crossentropy(
            self.probabilities, self.targets).mean() +
                      3e-5 * (self.l2cost + self.l1cost + self.l21cost))

        self.classification = T.argmax(self.probabilities, axis=1)

        self._grads = T.grad(self._cost, self.params)

        self._classify = theano.function(
            [self.inputs], self.classification)
        self._get_probabilities = theano.function(
            [self.inputs], self.probabilities)
        

    def get_probabilities(self, inputs):
        # switch to test mode
        self.mode.set_value(np.int8(1))
        probs = np.zeros((inputs.shape[0], 7), dtype=np.float32)
        for b in range(inputs.shape[0] // self.batchsize):
            #print '{0}/{1}'.format(b+1, inputs.shape[0] // self.batchsize)
            probs[
                b * self.batchsize:
                (b + 1) * self.batchsize] = self._get_probabilities(
                inputs[b * self.batchsize: (b + 1) * self.batchsize])
        if inputs.shape[0] % self.batchsize:
            last_batch = np.zeros((self.batchsize, inputs.shape[1]),
                                  dtype=theano.config.floatX)
            last_batch[:(inputs.shape[0] % self.batchsize)] = \
                inputs[-(inputs.shape[0] % self.batchsize):]
            probs[-(inputs.shape[0] % self.batchsize):] = self._get_probabilities(
                last_batch)[:(inputs.shape[0] % self.batchsize)]
        # switch to train mode
        self.mode.set_value(np.int8(0))

        return probs
        

    def classify(self, inputs):
        # switch to test mode
        self.mode.set_value(np.int8(1))
        rval = self._classify(inputs)
        # switch to train mode
        self.mode.set_value(np.int8(0))
        return rval
        

    def compute_01_loss(self, inputs, targets, batchsize=128):
        # switch to test mode
        self.mode.set_value(np.int8(1))
        predictions = np.zeros((inputs.shape[0],), dtype=np.int32)
        for b in range(inputs.shape[0] // batchsize):
            #print '{0}/{1}'.format(b+1, inputs.shape[0] // batchsize)
            predictions[
                b * batchsize:
                (b + 1) * batchsize] = self.classify(
                inputs[b * batchsize: (b + 1) * batchsize])
        if inputs.shape[0] % batchsize:
            last_batch = np.zeros((batchsize, inputs.shape[1]),
                                  dtype=theano.config.floatX)
            last_batch[:(inputs.shape[0] % batchsize)] = \
                inputs[-(inputs.shape[0] % batchsize):]
            predictions[-(inputs.shape[0] % batchsize):] = self.classify(
                last_batch)[:(inputs.shape[0] % batchsize)]
        # switch to train mode
        self.mode.set_value(np.int8(0))

        return np.sum(targets != predictions) / np.float32(inputs.shape[0])

# vim: set ts=4 sw=4 sts=4 expandtab:
