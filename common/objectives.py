#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T

def Frobenius(W):
    return (W**2).sum()**.5

def L2_sqr(W):
    return (W**2).sum()

def L21(W):
    return ((W**2).sum(1)**.5).sum(0)


def L11(W):
    return abs(W).sum()


# vim: set ts=4 sw=4 sts=4 expandtab:
