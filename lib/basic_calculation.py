#!/usr/bin/env python
# coding: utf8

import os
import sys
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import yaml

# -sys.float_info.max ~ -1E300

rng = np.random.default_rng()

#def set_seed(seed):
#    rng = np.random.default_rng(seed)

def normalize_ndarray(vd):
    total = np.sum(vd)
    if total == 0:
        raise ValueError("Erro: sum is 0.")
    return vd / total

def log_normalize_ndarray(vd):
    vd = vd - np.max(vd)
    log_total = np.log( np.sum( np.exp(vd) ) )
    return vd - log_total

def logadd(a, b):
    if a > b:
        return a+math.log(1+math.exp(b-a))
    else:
        return b+math.log(1+math.exp(a-b))

def sample_distr(np_array):
    val = rng.random()
    for i in range(np_array.shape[0]):
        if val < np_array[i]:
            return i
        else:
            val -= np_array[i]
    return np_array.shape[0] - 1


class Prob:

    def __init__(self, dim=None):
        if dim is not None:
            self.P = np.ones(dim) / dim
            self.LP = np.zeros(dim) - np.log(dim)

    def __str__(self):
        return f"{self.P}"
#        return f"{self.P}\n{self.LP}"

    def P2LP(self):
        self.LP = np.array([math.log(x) for x in self.P])

    def LP2P(self):
        self.P = np.array([math.exp(x) for x in self.LP])

    def normalize(self):
        self.P = normalize_ndarray(self.P)
        self.P2LP()

    def log_normalize(self):
        self.LP = log_normalize_ndarray(self.LP)
        self.LP2P()

    def randomize(self):
        self.P = rng.random(self.P.shape[0])
        self.normalize()

    def change_temperature(self, inv_temprature):
        for i in range(self.P.shape[0]):
            self.P[i] = pow(self.P[i], inv_temprature)
        self.normalize()


