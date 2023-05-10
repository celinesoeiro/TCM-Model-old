"""
@author: Celine Soeiro

@description: Functions that are not necessarily part of the model but are used 
to obtain the desired result

- randn -> Obtain normally distributed random numbers
- poissonSpikeGen -> Poissonian firing frequency from other parts of the brain
- pink Noise Generation -> Author: Ph.D. Eng. Hristo Zhivomirov 

"""

import numpy as np
import math
from scipy.fft import fft, ifft

def poissonSpikeGen(fr, tSim, nTrials, dt):    
    nBins = int(np.floor(tSim/dt))
    spikeMat = np.random.rand(nTrials, nBins) < fr*dt
    tVec = np.arange(0,tSim - dt, dt)
    
    return spikeMat, tVec

def pinkNoise(m, n):
# =============================================================================
#   m - number of matrix rows
#   n - number of matrix columns
#   x - returned column vector of pink noise samples with unity standard deviation and 
# zero mean value 
#
#  The function generates a column vector of pink (flicker) noise samples. In 
# terms of power at a constant bandwidth, the pink noise falls off by -3 dB/oct 
# i.e., -10 dB/dec.
# =============================================================================
    # Define the length of the noise vector and ensure that M is even, this will simplify the processing
    m = np.round(m); n = np.round(n); N = m*n;
    if (np.remainder(N,2)):
        M = N + 1
    else:
        M = N
    # set the PSD slope
    alpha = -1
    # convert from PSD (power specral density) slope to ASD (amplitude spectral density) slope
    alpha = alpha/2
    # generate AWGN signal
    x = np.random.randn(1, M)
    # calculate the number of unique Fast Fourier Transform points
    Num_Unique_Pts = math.ceil((M + 1)/2)
    # take Fast Fourier Transform of x
    X = fft(x)
    # fft is symmetric, throw away the second half
    X = X[0,Num_Unique_Pts]
    # prepare a vector with frequency indexes 
    f = np.arange(1,Num_Unique_Pts)
    # manipulate the left half of the spectrum so the spectral 
    # amplitudes are proportional to the frequency by factor f^alpha
    X = X*(f**int(alpha))
    # perform ifft
    x = ifft(X)
    # ensure zero mean value and unity standard deviation 
    x = x - np.mean(x)
    x = x/x.std(axis=0)
    x = x.flatten()
    return x



    