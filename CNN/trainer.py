import collections
import warnings

import numpy as np
import theano
import theano.tensor as T


class SGD_Trainer(object):
    """Implementation of a stochastic gradient descent trainer
    """

#{{{ Properties

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, val):
        #FIXME: make this work for other input types
        if not isinstance(val, np.ndarray):
            raise TypeError(
                'Resetting trainer inputs currently only works for '
                'ndarray inputs!')
        self._inputs = val
        self._inputs_theano = theano.shared(
            self._inputs[:self.loadsize],
            name='inputs')
        self.numcases = self._inputs.shape[0]
        print 'recompiling trainer functions...'
        self._compile_functions()

    @property
    def gradient_clip_threshold(self):
        return self._gradient_clip_threshold.get_value()

    @gradient_clip_threshold.setter
    def gradient_clip_threshold(self, val):
        self._gradient_clip_threshold.set_value(np.float32(val))

    @property
    def learningrate(self):
        return self._learningrate.get_value()

    @learningrate.setter
    def learningrate(self, value):
        self._learningrate.set_value(np.float32(value))

    @property
    def momentum(self):
        return self._momentum.get_value()

    @momentum.setter
    def momentum(self, val):
        self._momentum.set_value(np.float32(val))

    @property
    def epochcount(self):
        return self._epochcount

    @epochcount.setter
    def epochcount(self, val):
        self._epochcount = int(val)

    @property
    def momentum_batchcounter(self):
        return self._momentum_batchcounter

    @property
    def nparams(self):
        return self._nparams
#}}}

    def __init__(self, model=None, inputs=None, outputs=None, masks=None,
                 batchsize=100, learningrate=.01,
                 momentum=0.9, loadsize=None,
                 rng=None, verbose=True,
                 gradient_clip_threshold=1000,
                 numepochs_per_load=1,
                 rmsprop=False, cost=None, params=None, inputvar=None,
                 grads=None, monitor_update_weight_norm_ratio=False,
                 auto_reset_on_naninf=True,
                 monitor_by_epoch_not_load=False,
                 targetvar=None, maskvar=None, preprocess_fn=None):

        self.verbose = verbose
        self.monitor_by_epoch_not_load = monitor_by_epoch_not_load
        if not hasattr(preprocess_fn, '__call__'):
            preprocess_fn = lambda x: x
        self.preprocess_fn = preprocess_fn


#{{{ Initialization of Properties
        assert model is not None or (
            cost is not None and params is not None and
            inputvar is not None and grads is not None
        ), (
            "either a model instance or cost, params and inputvar "
            "have to be passed to the SGD_Trainer constructor")

        self.auto_reset_on_naninf = auto_reset_on_naninf
        self.monitor_update_weight_norm_ratio = monitor_update_weight_norm_ratio
        print 'monitor_update_weight_norm_ratio: {0}'.format(monitor_update_weight_norm_ratio, )

        self.is_supervised = False
        self.has_masks = False

        self._inputs_type = 'numpy'
        self._inputs = inputs
        if outputs is not None:
            self._outputs = outputs
            self.is_supervised = True
        if masks is not None:
            self.has_masks = True
            self._masks = masks

        if model is not None:
            self._model = model
            self._params = model.params
            self._cost = model._cost
            self._inputvar = model.inputs
            self._grads = model._grads
            if self.is_supervised:
                try:
                    self._targetvar = model.targets
                except AttributeError:
                    print 'Training in supervised mode, but model does not have target variable'
                    raise
            if self.has_masks:
                try:
                    self._maskvar = model.masks
                except AttributeError:
                    print 'Training in masked cost mode, but model does not have mask variable'
                    raise
        else:
            self._params = params
            self._cost = cost
            self._inputvar = inputvar
            self._grads = grads
            if self.is_supervised:
                try:
                    self._targetvar = targetvar
                except AttributeError:
                    print 'Training in supervised mode, but no target variable specified'
                    raise
            if self.has_masks:
                try:
                    self._maskvar = maskvar
                except AttributeError:
                    print 'Training in masked cost mode, but no mask variable specified'
                    raise

        # compute total number of params
        self._nparams = 0
        for p in self._params:
            try:
                self._nparams += p.get_value().size
            except AttributeError:
                # handles scalar params
                self._nparams += 1
        print 'number of params: {0}'.format(self._nparams)

        if monitor_update_weight_norm_ratio:
            self._update_weight_norm_ratios_log = dict(
                [(p, []) for p in self._params])

        self._learningrate = theano.shared(np.float32(learningrate),
                                           name='learningrate')
        self.numepochs_per_load = numepochs_per_load

        self._momentum = theano.shared(np.float32(momentum),
                                       name='momentum')
        self._total_stepcount = 0

        self._gradient_clip_threshold = theano.shared(
            np.float32(gradient_clip_threshold),
            name='gradient_clip_threshold')
        self._avg_gradnorm = theano.shared(np.float32(0.), name='avg_gradnorm')

        self._model = model

        self._numparams = reduce(lambda x, y: x + y,
                                 [p.get_value().size for p in self._params])

        self.numcases = self._inputs.shape[0]

        self.batchsize = batchsize
        self.loadsize = loadsize
        if self.batchsize > self.loadsize:
            warnings.warn('batchsize > loadsize -> batchsize := loadsize')
            self.batchsize = self.loadsize

        if self.loadsize % self.batchsize:
            self.loadsize = int(
                self.loadsize // self.batchsize + 1) * self.batchsize

        if rng is None:
            self._rng = np.random.RandomState(1)
        else:
            self._rng = rng

        # prepare loads
        self.prepare_loads()

        # use first load to allocate shared variable
        self._inputs_theano = theano.shared(
            self._inputs[self.data_indices[0]],
            name='inputs')
        if self.is_supervised:
            self._outputs_theano = theano.shared(
                self._outputs[self.data_indices[0]],
                name='outputs')
        if self.has_masks:
            self._masks_theano = theano.shared(
                self._masks[self.data_indices[0]],
                name='masks')

        self._momentum_batchcounter = 0

        self._epochcount = 0
        self._index = T.lscalar()
        self._incs = \
            dict([(p, theano.shared(
                value=np.zeros(p.get_value().shape,
                               dtype=theano.config.floatX),
                name='inc_' + p.name))
                for p in self._params])
        self._inc_updates = collections.OrderedDict()
        self.rmsprop = rmsprop
        if self.rmsprop:
            self.averaging_coeff = 0.95
            self.stabilizer = 1e-2
            self._avg_grad_sqrs = \
                dict([(p, theano.shared(value=np.zeros(
                    p.get_value().shape,
                    dtype=theano.config.floatX), name='avg_grad_sqr_' + p.name))
                    for p in self._params])
        self._avg_grad_sqrs_updates = collections.OrderedDict()
        self._updates_nomomentum = collections.OrderedDict()
        self._updates = collections.OrderedDict()
        self._n = T.lscalar('n')
        self._n.tag.test_value = 0.
        self._noop = 0.0 * self._n
        self._batch_idx = theano.shared(
            value=np.array(0, dtype=np.int64), name='batch_idx')

        self.costs = []
        self._compile_functions()

#}}}

    def prepare_loads(self):
        # rand permutation of training cases
        perm = self._rng.permutation(self.numcases)
        # compute num of cases to duplicate to fill last batch to batchsize
        n_duplicates = self.numcases % self.loadsize

        if n_duplicates:
            warnings.warn('randomly duplicating some samples to fill last load')
            self.data_indices = np.concatenate((
                perm,
                self._rng.randint(
                    self.numcases, size=self.loadsize - n_duplicates)),
                axis=0)
        else:
            self.data_indices = perm
        self.data_indices = self.data_indices.reshape(-1, self.loadsize)
        self.numloads = self.data_indices.shape[0]

    def __del__(self):
        if self._inputs_type == 'h5':
            self._inputsfile.close()

    def reset_incs(self):
        for p in self._params:
            self._incs[p].set_value(
                np.zeros(p.get_value().shape, dtype=theano.config.floatX))

    def reset_avg_grad_sqrs(self):
        if self.rmsprop:
            for p in self._params:
                self._avg_grad_sqrs[p].set_value(
                    np.zeros(p.get_value().shape, dtype=theano.config.floatX))

    def _compile_functions(self):
        self._gradnorm = T.zeros([])
        for _param, _grad in zip(self._params, self._grads):
            # apply rmsprop to before clipping gradients
            if self.rmsprop:
                avg_grad_sqr = self._avg_grad_sqrs[_param]
                new_avg_grad_sqr = self.averaging_coeff * avg_grad_sqr + \
                    (1 - self.averaging_coeff) * T.sqr(_grad)
                self._avg_grad_sqrs_updates[avg_grad_sqr] = new_avg_grad_sqr
                rms_grad_t = T.sqrt(new_avg_grad_sqr)
                rms_grad_t = T.maximum(rms_grad_t, self.stabilizer)
                _grad = _grad / rms_grad_t
            self._gradnorm += T.sum(_grad**2)
        self._gradnorm = T.sqrt(self._gradnorm)

        self._givens = {}
        self._givens[self._inputvar] = self._inputs_theano[
            self._batch_idx * self.batchsize:
            (self._batch_idx + 1) * self.batchsize]
        if self.is_supervised:
            self._givens[self._targetvar] = self._outputs_theano[
                self._batch_idx * self.batchsize:
                (self._batch_idx + 1) * self.batchsize]
        if self.has_masks:
            self._givens[self._maskvar] = self._masks_theano[
                self._batch_idx * self.batchsize:
                (self._batch_idx + 1) * self.batchsize]

        self.gradnorm = theano.function(
            inputs=[],
            outputs=self._gradnorm,
            givens=self._givens)

        avg_gradnorm_update = {
            self._avg_gradnorm: self._avg_gradnorm * .8 + self._gradnorm * .2}

        self._update_weight_norm_ratios = []
        for _param, _grad in zip(self._params, self._grads):
            if hasattr(self._model, 'skip_params'):
                if _param.name in self._model.skip_params:
                    continue

            _clip_grad = T.switch(
                T.gt(self._gradnorm, self._gradient_clip_threshold),
                _grad * self._gradient_clip_threshold / self._gradnorm, _grad)

            try:  # ... to apply learningrate_modifiers
                # Cliphid version:
                self._inc_updates[self._incs[_param]] = \
                    self._momentum * self._incs[_param] - \
                    self._learningrate * \
                    self._model.layer.learningrate_modifiers[
                        _param.name] * _clip_grad

                self._updates[_param] = _param + self._incs[_param]
                self._updates_nomomentum[_param] = _param - \
                    self._learningrate * \
                    self._model.layer.learningrate_modifiers[_param.name] * \
                        _clip_grad

            except AttributeError:
                self._inc_updates[self._incs[_param]] = self._momentum * \
                        self._incs[_param] - self._learningrate * _clip_grad
                self._updates[_param] = _param + self._incs[_param]
                self._updates_nomomentum[_param] = _param - \
                        self._learningrate * _clip_grad

            if self.monitor_update_weight_norm_ratio:
                print 'building update weight norm ratio graph for ', _param.name
                self._update_weight_norm_ratios.append(
                    T.mean(self._incs[_param]**2) / T.mean(
                        _param**2))


        # compute function to get update_weight_norm_ratios (returned in same
        # order as params list)
        print 'compiling update weight norm ratio function'
        self.get_update_weight_norm_ratios = theano.function(
            [], self._update_weight_norm_ratios)
        print 'done'

        # first update gradient norm running avg
        ordered_updates = collections.OrderedDict()
        try:
            ordered_updates.update(self._model.updates)
        except AttributeError:
            pass
        ordered_updates.update(avg_gradnorm_update)
        # so that it is considered in the parameter update computations
        ordered_updates.update(self._inc_updates)
        self._updateincs = theano.function(
            [], [self._cost, self._avg_gradnorm], updates = ordered_updates,
            givens = self._givens)

        self._trainmodel = theano.function(
            [self._n], self._noop, updates = self._updates)

        self._trainmodel_nomomentum = theano.function(
            [self._n], self._noop, updates = self._updates_nomomentum,
            givens = self._givens)

        self._momentum_batchcounter = 0


    def _trainsubstep(self, batchidx):
        self._batch_idx.set_value(batchidx)
        stepcost, avg_gradnorm = self._updateincs()
        # catch NaN, before updating params
        try:
            if np.isnan(stepcost):
                raise ValueError, 'Cost function returned nan!'
            elif np.isinf(stepcost):
                raise ValueError, 'Cost function returned infinity!'
        except ValueError:
            if self.auto_reset_on_naninf:
                print 'nan or inf detected, resetting...'
                self.reset_incs()
                self.reset_avg_grad_sqrs()
                self._avg_gradnorm.set_value(0.0)
            else:
                print ('nan or inf detected, auto_reset_on_naninf is set to '
                       'False. Set it to True to automagically reset the '
                       'trainer and continue training.')
            raise

        if self._momentum_batchcounter < 10:
            self._momentum_batchcounter += 1
            self._trainmodel_nomomentum(0)
        else:
            self._momentum_batchcounter = 10
            self._trainmodel(0)
        return stepcost, avg_gradnorm

    def get_avg_gradnorm(self):
        avg_gradnorm = 0.0
        num_batches = self.loadsize//self.batchsize
        print self.gradnorm()
        for batch_idx in range(num_batches):
            self._batch_idx.set_value(batch_idx)
            tmp = self.gradnorm()
            avg_gradnorm += tmp / num_batches
        print avg_gradnorm
        return avg_gradnorm

    def upload_data(self, loadidx):
        # upload preprocessed data
        self._inputs_theano.set_value(
            self.preprocess_fn(self._inputs[self.data_indices[loadidx]]))
        if self.is_supervised:
            self._outputs_theano.set_value(
                self._outputs[self.data_indices[loadidx]])
        if self.has_masks:
            self._masks_theano.set_value(
                self._masks[self.data_indices[loadidx]])

    def step(self):
        cost = 0.0
        stepcount = 0.0
        epoch_gradnorm = 0.0

        self._epochcount += 1
        self.prepare_loads()
        for load_index in range(self.numloads):
            self.upload_data(load_index)

            for batch_index in range(self.loadsize//self.batchsize):
                stepcount += 1.0
                self._total_stepcount += 1.0
                stepcost, avg_gradnorm = self._trainsubstep(batch_index)
                cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)* \
                    stepcost
                epoch_gradnorm = (1.0-1.0/stepcount)*epoch_gradnorm + (1.0/stepcount)* \
                    avg_gradnorm
            if self.verbose and not self.monitor_by_epoch_not_load:
                print '> epoch {0:d}, load {1:d}/{2:d}, cost: {3:f}, avg. gradnorm: {4}'.format(
                    self._epochcount, load_index + 1, self.numloads, cost, avg_gradnorm)
                if hasattr(self._model, 'monitor'):
                    self._model.monitor()
            if self.monitor_update_weight_norm_ratio and not self.monitor_by_epoch_not_load: # TODO: monitor_by_epoch cases
                print 'computing update weight norm ratios of last random batch'
                ratios = self.get_update_weight_norm_ratios()
                print 'len(ratios): {0}'.format(len(ratios), )
                for p, ratio in zip(self._params, ratios):
                    self._update_weight_norm_ratios_log[p].append(ratio)
                    if self.verbose:
                        print p.name, 'update/weight norm ratio: ', ratio

        self.costs.append(cost) # TODO: checkme
        if self.verbose and self.monitor_by_epoch_not_load:
            print '> epoch {0:d}, cost: {1:f}, avg. gradnorm: {2}'.format(
                self._epochcount, cost, epoch_gradnorm)
            if hasattr(self._model, 'monitor'):
                self._model.monitor()

        return cost
