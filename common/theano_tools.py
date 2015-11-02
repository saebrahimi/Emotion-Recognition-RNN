#!/usr/bin/env python
#-*- coding: utf-8 -*-

import theano
import theano.tensor as T

# define some symbolic variables
theano_matrix1 = T.matrix(name='theano_matrix1')
theano_matrix2 = T.matrix(name='theano_matrix2')

# define some functions

# dot product/matrix product
theano_dot = theano.function([theano_matrix1, theano_matrix2], T.dot(
    theano_matrix1, theano_matrix2), name='theano_dot')

theano_scalar = T.fscalar(name='theano_scalar')
theano_scale = theano.function(
    [theano_matrix1, theano_scalar], theano_matrix1 * theano_scalar, name='scale')

# elementwise product
theano_multiply = theano.function(
    [theano_matrix1, theano_matrix2], theano_matrix1 * theano_matrix2, name='theano_multiply')

theano_row_vector = T.row(name='theano_row_vector')
theano_col_vector = T.col(name='theano_col_vector')

theano_subtract_row = theano.function(
    [theano_matrix1, theano_row_vector], theano_matrix1 - theano_row_vector, name='theano_subtract_row')
theano_divide_row = theano.function(
    [theano_matrix1, theano_row_vector], theano_matrix1 / theano_row_vector, name='theano_divide_row')
theano_subtract_col = theano.function(
    [theano_matrix1, theano_col_vector], theano_matrix1 - theano_col_vector, name='theano_subtract_col')
theano_divide_col = theano.function(
    [theano_matrix1, theano_col_vector], theano_matrix1 / theano_col_vector, name='theano_divide_col')

theano_var1 = theano.function(
    [theano_matrix1], T.var(theano_matrix1, 1), name='theano_var1')
theano_mean0 = theano.function(
    [theano_matrix1], T.mean(theano_matrix1, 0), name='theano_mean0')
theano_mean1 = theano.function(
    [theano_matrix1], T.mean(theano_matrix1, 1), name='theano_mean1')

_theano_ssd0 = T.sum((theano_matrix1 - theano_row_vector) ** 2, 0)
theano_ssd0 = theano.function(
    [theano_matrix1, theano_row_vector], _theano_ssd0, name='ssd0')
theano_sqrt = theano.function(
    [theano_matrix1], T.sqrt(theano_matrix1), name='sqrt')

theano_sum_of_squares = theano.function(inputs=[theano_matrix1], outputs=T.sum(
    theano_matrix1 ** 2, axis=1), name='theano_sum_of_squares')



# vim: set ts=4 sw=4 sts=4 expandtab:
