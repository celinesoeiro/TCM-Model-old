"""
@author: Celine Soeiro

@description: Functions that are not necessarily part of the model but are used 
to obtain the desired result

- randn -> Obtain normally distributed random numbers
- poissonSpikeGen -> Poissonian firing frequency from other parts of the brain
- 

"""

import numpy as np
from scipy.stats import norm

def randn(number, column = False):
    np.random.seed(number)
    if (column):
        return norm.ppf(np.random.rand(number,1))
    else:
        return norm.ppf(np.random.rand(1,number)) 


def poissonSpikeGen(fr, tSim, nTrials, dt):    
    nBins = int(np.floor(tSim/dt))
    spikeMat = np.random.rand(nTrials, nBins) < fr*dt
    tVec = np.arange(0,tSim - dt, dt)
    
    return spikeMat, tVec