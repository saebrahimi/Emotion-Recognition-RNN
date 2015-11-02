import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal.downsample import max_pool_2d

import numpy as np


class Layer(object):
    def __init__(self, inputs, inputs_shape=None):
        """
        Useful to get outputs shape
        """

        if isinstance(inputs, Layer):
            self.inputs = inputs.outputs
            self.inputs_shape = inputs.outputs_shape
        else:
            assert(inputs_shape is not None)
            self.inputs = inputs
            self.inputs_shape = inputs_shape
        print '({0}) input shape {1}'.format(self.name, self.inputs_shape)


class RandFlip(Layer):
    def __init__(self, inputs, image_shape, name, theano_rng, mode_var):
        # one window for whole batch
        self.name = name
        super(RandFlip, self).__init__(inputs, image_shape)

        self.theano_rng = theano_rng
        self.mode = mode_var

        self.params = []

        self.flipped = self.inputs[:,:,:,::-1]
        self.doflip = self.theano_rng.binomial(
            n=1, p=.5, size=(self.inputs.shape[0],))

        # if train mode: randomly flip, else don't flip
        self.outputs = T.switch(
            self.mode,
            self.inputs,
            T.switch(self.doflip.dimshuffle(0,'x','x','x'),
                     self.flipped, self.inputs))
        #self.outputs = inputs
        self.outputs_shape = self.inputs_shape


class ConvLayer(Layer):
    def __init__(self, rng, inputs, filter_shape,
                 name, image_shape=None, pad=0, init_scale=None):
        """
        Convolutional layer

        Args
        ----
        rng: instance of numpy.random.RandomState
        inputs: symbolic theano variable
        filter_shape: tuple of 4 ints (channels_out)
        """

        self.name = name
        super(ConvLayer, self).__init__(inputs, image_shape)
        assert self.inputs_shape[1] == filter_shape[1]

        self.rng = rng
        self.filter_shape = filter_shape
        self.pad = pad

        if init_scale is None:
            # if we don't specify a scale for weight initialization,
            # we use the formula
            # 1/sqrt(number of weights in each filter)
            init_scale = 1. / np.sqrt(
                filter_shape[1] *
                filter_shape[2] *
                filter_shape[3])

        self.init_scale = init_scale

        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-self.init_scale, high=self.init_scale,
                            size=self.filter_shape),
                dtype=theano.config.floatX
            ), name='{0}_W'.format(self.name)
        )

        self.params = [self.W]

        # if padding is greater than zero, we insert the inputs into
        # the center of a larger zero array, effectively adding zero
        # borders
        if self.pad > 0:
            self.padded_inputs = T.set_subtensor(
                T.zeros((self.inputs_shape[0],
                         self.inputs_shape[1],
                         self.inputs_shape[2] + 2 * self.pad,
                         self.inputs_shape[3] + 2 * self.pad),
                        dtype=self.inputs.dtype)[:, :, pad:-pad, pad:-pad],
                self.inputs
            )
        else:
            self.padded_inputs = self.inputs
        self.padded_inputs_shape = (
            self.inputs_shape[0],
            self.inputs_shape[1],
            self.inputs_shape[2] + 2 * self.pad,
            self.inputs_shape[3] + 2 * self.pad)

        self.outputs = conv.conv2d(
            input=self.padded_inputs,
            filters=self.W,
            filter_shape=self.filter_shape,
            image_shape=self.padded_inputs_shape
        )

        self.outputs_shape = (
            self.inputs_shape[0], self.filter_shape[0],
            self.inputs_shape[2] + 2 * self.pad - self.filter_shape[2] + 1,
            self.inputs_shape[3] + 2 * self.pad - self.filter_shape[3] + 1)


class MaxPoolLayer(Layer):
    def __init__(self, inputs, pool_size, name, ignore_border=True, stride=None):
        """
        Max pooling layer

        """

        self.name = name
        super(MaxPoolLayer, self).__init__(inputs)
        self.pool_size = pool_size
        self.ignore_border = ignore_border
        if stride is None:
            stride = pool_size
        self.stride = stride

        self.params = []

        self.outputs = max_pool_2d(
            input=self.inputs,
            ds=self.pool_size,
            ignore_border=self.ignore_border,
            st=self.stride
        )

        self.outputs_shape = (
            self.inputs_shape[0],
            self.inputs_shape[1],
            #int(np.ceil((self.inputs_shape[2] - pool_size[0]) / np.float32(stride[0]))) + 1,
            #int(np.ceil((self.inputs_shape[3] - pool_size[1]) / np.float32(stride[1]))) + 1
            (self.inputs_shape[2] - pool_size[0]) // stride[0] + 1,
            (self.inputs_shape[3] - pool_size[1]) // stride[1] + 1
        )


class ConvBiasLayer(Layer):
    def __init__(self, inputs, name):
        """
        Add bias
        """

        self.name = name
        super(ConvBiasLayer, self).__init__(inputs)

        self.b = theano.shared(
            np.zeros(
                (self.inputs_shape[1],), dtype=theano.config.floatX
            ), name='{0}_b'.format(self.name)
        )

        self.params = [self.b]

        self.outputs = self.inputs + self.b.dimshuffle('x', 0, 'x', 'x')

        self.outputs_shape = self.inputs_shape


class RandCropAndFlip(Layer):
    def __init__(self, inputs, image_shape, patch_size, name, theano_rng, mode_var):
        # one window for whole batch
        self.name = name
        super(RandCropAndFlip, self).__init__(inputs, image_shape)

        self.patch_size = patch_size
        self.theano_rng = theano_rng
        self.mode = mode_var

        self.params = []

        print 'self.inputs_shape: {0}'.format(self.inputs_shape, )
        print 'patch_size: {0}'.format(patch_size, )
        print 'self.inputs_shape[2] - patch_size[0]: {0}'.format(self.inputs_shape[2] - patch_size[0], )
        print 'self.inputs_shape[3] - patch_size[1]: {0}'.format(self.inputs_shape[2] - patch_size[0], )
        self.rand_row_coord = self.theano_rng.random_integers(
            low=0, high=self.inputs_shape[2] - patch_size[0])
        self.rand_col_coord = self.theano_rng.random_integers(
            low=0, high=self.inputs_shape[3] - patch_size[1])

        self.center_row_coord = (self.inputs.shape[2] - patch_size[0]) // 2
        self.center_col_coord = (self.inputs.shape[3] - patch_size[1]) // 2

        self.row_coord = T.switch(
            self.mode, self.center_row_coord, self.rand_row_coord)
        self.col_coord = T.switch(
            self.mode, self.center_col_coord, self.rand_col_coord)

        self.patches = self.inputs[:,
                                   :,
                                   self.row_coord:self.row_coord + patch_size[0],
                                   self.col_coord:self.col_coord + patch_size[1]]

        self.flipped = self.patches[:,:,:,::-1]
        self.doflip = self.theano_rng.binomial(
            n=1, p=.5, size=(self.inputs.shape[0],))

        # if train mode: randomly flip, else don't flip
        self.outputs = T.switch(
            self.mode,
            self.patches,
            T.switch(self.doflip.dimshuffle(0,'x','x','x'), self.flipped, self.patches))

        self.outputs_shape = self.inputs_shape[:2] + self.patch_size


class Dropout(Layer):
    def __init__(self, inputs, dropout_rate, name, theano_rng, mode_var, inputs_shape=None):
        """Dropout

        Args
        ----
        mode_var: symbolic variable, which has value 0 during training and 1
            during test time
        """

        self.name = name
        super(Dropout, self).__init__(inputs, inputs_shape)

        self.dropout_rate = theano.shared(
            dropout_rate, '{0}_dropout_rate'.format(name))
        self.theano_rng = theano_rng
        self.mode = mode_var

        self.params = []

        self.mask = self.theano_rng.binomial(
            n=1, p=1 - dropout_rate, size=self.inputs_shape,
            dtype=theano.config.floatX)

        self.outputs = T.switch(self.mode,
                                self.inputs * (1. - dropout_rate),
                                self.inputs * self.mask)

        self.outputs_shape = self.inputs_shape


class AffineLayer(Layer):
    def __init__(self, rng, inputs, nouts,
                 name, init_scale=None, nins=None, with_bias=True,
                 inputs_shape=None):
        """
        Fully connected layer with bias option

        Args
        ----
        rng: instance of numpy.random.RandomState
        """

        self.name = name
        super(AffineLayer, self).__init__(inputs, inputs_shape=inputs_shape)
        self.rng = rng
        self.nins = self.inputs_shape[-1]
        self.nouts = nouts

        if init_scale is None:
            # if we don't specify a scale for weight initialization,
            # we use the formula
            # 1/sqrt(number of weights in each filter)
            init_scale = 1. / np.sqrt(self.nins)

        self.init_scale = init_scale
        self.with_bias = with_bias

        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-self.init_scale, high=self.init_scale,
                            size=(self.nins, self.nouts)),
                dtype=theano.config.floatX
            ), name='{0}_W'.format(self.name)
        )

        self.params = [self.W]

        self.outputs = T.dot(self.inputs, self.W)

        if with_bias:
            self.b = theano.shared(
                np.zeros(
                    self.nouts, dtype=theano.config.floatX
                ), name='{0}_b'.format(self.name)
            )
            self.params.append(self.b)
            self.outputs = self.outputs + self.b

        self.outputs_shape = (self.inputs_shape[0], self.nouts)


class Relu(Layer):
    def __init__(self, inputs, name):
        """
        Relu activation function
        """

        self.name = name
        super(Relu, self).__init__(inputs)

        self.params = []

        self.outputs = T.switch(self.inputs < 0, 0, self.inputs)
        self.outputs_shape = self.inputs_shape


class RectifiedTanh(Layer):
    def __init__(self, inputs, name):
        """
        Relu activation function
        """

        self.name = name
        super(RectifiedTanh, self).__init__(inputs)

        self.params = []

        self.outputs = T.switch(self.inputs < 0, 0, T.tanh(self.inputs))
        self.outputs_shape = self.inputs_shape


class Concat(Layer):
    def __init__(self, inputs_list, name, axis,
                 inputs_shape_list=None):
        """
        Concatenation layer
        """

        self.name = name
        self.axis = axis
        if inputs_shape_list is None:
            inputs_shape_list = [None] * len(inputs_list)
        assert len(inputs_shape_list) == len(inputs_list)

        self.inputs = []
        self.inputs_shape = []
        for i, inp in enumerate(inputs_list):        
            if isinstance(inp, Layer):
		self.inputs.append(inp.outputs)
                self.inputs_shape.append(inp.outputs_shape)
            else:
                assert(inputs_shape_list[i] is not None)
                self.inputs.append(inp)
                self.inputs_shape.append(inputs_shape_list[i])

        self.params = []

        # concatenate the inputs
        self.outputs = T.concatenate(self.inputs, axis=axis)

        self.outputs_shape = list(self.inputs_shape[0])
        for i in range(1, len(self.inputs_shape)):
            self.outputs_shape[axis] += self.inputs_shape[i][axis]


class Softmax(Layer):
    def __init__(self, inputs, name):
        """
        Softmax
        """

        self.name = name
        super(Softmax, self).__init__(inputs)

        self.params = []

        self.outputs = T.nnet.softmax(self.inputs)
        self.outputs_shape = self.inputs_shape


class Sigmoid(Layer):
    def __init__(self, inputs, name):
        """
        Sigmoid
        """

        self.name = name
        super(Sigmoid, self).__init__(inputs)

        self.params = []

        self.outputs = T.nnet.sigmoid(self.inputs)
        self.outputs_shape = self.inputs_shape


class Reshape(Layer):
    def __init__(self, inputs, shape, name):
        """
        Reshaping
        """

        self.name = name
        super(Reshape, self).__init__(inputs)

        self.params = []

        assert(np.prod(self.inputs_shape) == np.prod(shape))

        self.outputs = T.reshape(self.inputs, shape)
        self.outputs_shape = shape


class Composite(Layer):
    def __init__(self, layers, name):
        """
        Collection of layers used in fusion
        """

        self.layers = layers
        self.name = name
        super(Composite, self).__init__(self.layers[0].inputs,
                                        self.layers[0].inputs_shape)
        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params)
        self.outputs = self.layers[-1].outputs
        self.outputs_shape = self.layers[-1].outputs_shape

