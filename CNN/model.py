#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import shutil
import warnings

import numpy as np
import tables
import theano

class Model(object):

    def __init__(self, name):
        if type(self) == Model:
            raise NotImplementedError('This base class should not be used directly')
        self.name = name

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = np.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum

    def updateparams_fromdict(self, newparams):
        #import ipdb; ipdb.set_trace()
        for p in self.params:
            try:
                p.set_value(newparams[p.name])
            except KeyError:
                print 'param {0} not in dict (will not be overwritten)'.format(
                    p.name)

    def get_params_dict(self):
        return dict([(p.name, p.get_value()) for p in self.params])

    def get_params(self):
        return np.concatenate(
            [p.get_value().flatten() for p in self.params])

    def save(self, filename):
        ext = os.path.splitext(filename)[1]
        if ext == '.h5':
            print 'saving h5 file'
            self.save_h5(filename)
        elif ext == '.npy':
            print 'saving npy file'
            self.save_npy(filename)
        elif ext == '.npz':
            print 'saving npz file'
            self.save_npz(filename)
        else:
            print 'unknown file extension: {}'.format(ext)

    def save_h5(self, filename):
        try:
            shutil.copyfile(filename, '{}_bak'.format(filename))
        except IOError:
            print 'could not make backup of model param file (which is normal if we haven\'t saved one until now)'

        with tables.openFile(filename, 'w') as h5file:
            for p in self.params:
                h5file.createArray(h5file.root, p.name, p.get_value())
                h5file.flush()

    def save_npy(self, filename):
        np.save(filename, self.get_params())

    def save_npz(self, filename):
        np.savez(filename, **(self.get_params_dict()))

    def load_h5(self, filename):
        h5file = tables.openFile(filename, 'r')
        new_params = {}
        for p in h5file.listNodes(h5file.root):
            new_params[p.name] = p.read()
        self.updateparams_fromdict(new_params)
        h5file.close()


    def load(self, filename):
        ext = os.path.splitext(filename)[1]
        if ext == '.h5':
            self.load_h5(filename)
        else:
            try:
                new_params = np.load(filename)
            except IOError, e:
                warnings.warn('''Parameter file could not be loaded with numpy.load()!
                            Is the filename correct?\n %s''' % (e, ))
            if type(new_params) == np.ndarray:
                print "loading npy file"
                self.updateparams(new_params)
            elif type(new_params) == np.lib.npyio.NpzFile:
                print "loading npz file"
                self.updateparams_fromdict(new_params)
            else:
                warnings.warn('''Parameter file loaded, but variable type not
                            recognized. Need npz or ndarray.''', Warning)


# vim: set ts=4 sw=4 sts=4 expandtab:
