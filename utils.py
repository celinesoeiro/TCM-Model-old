# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 20:56:09 2022

@author: celin
"""

import numpy as np
from scipy.stats import norm

def randn(number, column = False, line = False):
    np.random.seed(number)
    if (column):
        return norm.ppf(np.random.rand(number,1))
    else:
        return norm.ppf(np.random.rand(1,number)) 


