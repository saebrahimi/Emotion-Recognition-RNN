import numpy as np
import theano
import theano.tensor as T

from model import Model


class RNN(Model):
    def __init__(self, nin, nout, nhid, numpy_rng, theano_rng, scale=1.0,
                 window_size=1, dropout_rate=0):
        self.nin = nin
        self.nout = nout
        self.nhid = nhid
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        self.scale = np.float32(scale)
        self.window_size = window_size
        self.dropout_rate = theano.shared(np.float32(dropout_rate), name='dropout_rate')

        self.inputs = T.fmatrix('inputs')
        self.targets = T.imatrix('targets')
        self.masks = T.bmatrix('masks')
        self.batchsize = self.inputs.shape[0]

        self.inputs_frames = self.inputs.reshape((
            self.batchsize, self.inputs.shape[1]/nin, nin)).dimshuffle(1,0,2)
        #self.targets_frames = self.targets.reshape((
        #    self.batchsize, self.targets.shape[1]/nout, nout)).dimshuffle(1,0,2)
        self.targets_frames = self.targets.T
        self.masks_frames = self.masks.T
        self.corrupted_frames = self.inputs_frames * (1./(
            1-self.dropout_rate)) * self.theano_rng.binomial(
            n=1, p=1-self.dropout_rate, size=self.inputs_frames.shape,
            dtype=theano.config.floatX)

        #self.h0_init = np.float32(.5) * np.ones(
        #    nhid, dtype=theano.config.floatX) + \
        #    numpy_rng.uniform(low=-.5, high=.5, size=nhid).astype(np.float32)
        #self.h0 = theano.shared(value=self.h0_init, name='h0')
        self.win = theano.shared(value=self.numpy_rng.normal(
            loc=0, scale=0.001, size=(nin*window_size, nhid)
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
            fn=self.step, sequences=T.arange(self.inputs_frames.shape[0] - self.window_size + 1),
            outputs_info=[T.zeros((self.batchsize, nhid), dtype=theano.config.floatX), None],
            non_sequences=[self.corrupted_frames])

        self.probabilities = T.nnet.softmax(self.outputs.reshape((
            self.outputs.shape[0] * self.outputs.shape[1],
            self.nout)))
        self.probabilities = T.clip(self.probabilities, 1e-6, 1-1e-6)

        self.get_hiddens = theano.function(
            [self.inputs], self.hiddens)
        self.get_probabilities = theano.function(
            [self.inputs], self.probabilities.reshape(self.outputs.shape))
            
        self._stepcosts = T.nnet.categorical_crossentropy(
            self.probabilities, self.targets_frames.flatten()).reshape(
                self.targets_frames.shape)

        self._cost = T.switch(T.gt(self.masks_frames, 0), self._stepcosts, 0).mean()
        self._grads = T.grad(self._cost, self.params)

        self._get_classifications = theano.function(
            [self.inputs], T.argmax(self.probabilities.reshape(self.outputs.shape), axis=2).T)
        
        self._get_classifications_seq = theano.function(
            [self.inputs], T.argmax(T.mean(self.probabilities.reshape(self.outputs.shape), axis=0), axis=1))
        
        self.max_probs = T.max(self.probabilities.reshape(self.outputs.shape), axis=2)
        self.classification_frame_onehot = T.eq(
            self.max_probs.dimshuffle(0, 1, 'x'),
            self.probabilities.reshape(self.outputs.shape)) * self.masks_frames.dimshuffle(0, 1, 'x')
        self._get_classifications_maxvote = theano.function(
            [self.inputs, self.masks], T.argmax(T.sum(self.classification_frame_onehot, axis=0), axis=1))
        
    def get_classifications(self, inputs):
        old_dropout_rate = self.dropout_rate.get_value()
        self.dropout_rate.set_value(np.float32(0))
        classifications = self._get_classifications(inputs)
        self.dropout_rate.set_value(old_dropout_rate)
        return classifications
    
    def get_classifications_seq(self, inputs):
        old_dropout_rate = self.dropout_rate.get_value()
        self.dropout_rate.set_value(np.float32(0))
        classifications = self._get_classifications_seq(inputs)
        self.dropout_rate.set_value(old_dropout_rate)
        return classifications
    
    def get_classifications_maxvote(self, inputs, masks):
        old_dropout_rate = self.dropout_rate.get_value()
        self.dropout_rate.set_value(np.float32(0))
        classifications = self._get_classifications_maxvote(inputs, masks)
        self.dropout_rate.set_value(old_dropout_rate)
        return classifications

    def step(self, t, h_tm1, corrupted_frames):
        inp_window_t = corrupted_frames[t:t+self.window_size].dimshuffle(1, 0, 2).reshape(
            (self.batchsize, self.window_size*self.nin))
        pre_h_t = T.dot(inp_window_t, self.win) + T.dot(h_tm1, self.wrnn)
        h_t = T.switch(pre_h_t > 0, pre_h_t, 0)
        o_t = T.dot(h_t, self.wout) + self.bout
        return h_t, o_t


