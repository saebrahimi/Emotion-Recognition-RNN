import numpy as np
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams
from model import Model


class RNN(Model):
    def __init__(self, nin, nout, nhid, numpy_rng, scale=1.0):
        self.nin = nin
        self.nout = nout
        self.nhid = nhid
        self.numpy_rng = numpy_rng
        self.theano_rng = RandomStreams(1)
        self.scale = np.float32(scale)

        self.inputs = T.fmatrix('inputs')
        self.targets = T.imatrix('targets')
        self.masks = T.bmatrix('masks')
        self.batchsize = self.inputs.shape[0]

        self.inputs_frames = self.inputs.reshape((
            self.batchsize, self.inputs.shape[1]/nin, nin)).dimshuffle(1,0,2)
        self.targets_frames = self.targets.T
        self.masks_frames = self.masks.T

        self.win = theano.shared(value=self.numpy_rng.normal(
            loc=0, scale=0.001, size=(nin, nhid)
        ).astype(theano.config.floatX), name='win')
        self.wrnn = theano.shared(value=self.scale * np.eye(
            nhid, dtype=theano.config.floatX), name='wrnn')
        self.wout = theano.shared(value=self.numpy_rng.uniform(
            low=-0.01, high=0.01, size=(nhid, nout)
        ).astype(theano.config.floatX), name='wout')
        self.bout = theano.shared(value=np.zeros(
            nout, dtype=theano.config.floatX), name='bout')

        self.params = [self.win, self.wrnn, self.wout, self.bout]

        (self.hiddens, self.outputs), self.updates = theano.scan(
            fn=self.step, sequences=self.inputs_frames,
            outputs_info=[self.theano_rng.uniform(low=0, high=1, size=(
                self.batchsize, nhid), dtype=theano.config.floatX), None])

        self.probabilities = T.nnet.softmax(self.outputs.reshape((
            self.outputs.shape[0] * self.outputs.shape[1],
            self.nout)))
        self.probabilities = T.clip(self.probabilities, 1e-6, 1-1e-6)

        self._stepcosts = T.nnet.categorical_crossentropy(
            self.probabilities, self.targets_frames.flatten()).reshape(
                self.targets_frames.shape)

        self._cost = T.switch(T.gt(self.masks_frames, 0), self._stepcosts, 0).mean()
        self._grads = T.grad(self._cost, self.params)

        self.get_classifications = theano.function(
            [self.inputs], T.argmax(self.probabilities.reshape(self.outputs.shape), axis=2).T)

    def step(self, inp_t, h_tm1):
        pre_h_t = T.dot(inp_t, self.win) + T.dot(h_tm1, self.wrnn)
        h_t = T.switch(pre_h_t > 0, pre_h_t, 0)
        o_t = T.dot(h_t, self.wout) + self.bout
        return h_t, o_t


