# -*- coding: utf-8 -*-
"""
Created on Thu May 19 22:23:57 2022

@author: celin
"""

import numpy as np

def dbsDelta(f_dbs, dbs_duration, dev, n_sim, Fs, cut):
    # This is to define Dirac delta pulses, no membrane current but straight dirac delta pulses that reach PNs:
    T_dbs = Fs/f_dbs
    dbs = np.arange(1,np.round(T_dbs), step=dbs_duration)
    I_dbs_full = np.zeros((1,dbs_duration))
    I_dbs_full[dbs] = 1 # Amplitude of DBS
    
    if (dev == 1):
        I_dbs = I_dbs_full;
    else:
        # extracellular dbs pulsing
        I_dbs = [np.zeros((1,cut)),
                 np.zeros((1,(n_sim-cut)/dev)),
                 I_dbs_full,
                 np.zeros((1,(n_sim-cut)/dev))]; 
    return I_dbs
    
    
def DBS(n_sim, synaptic_fidelity, Fs, chop_till):
    I_dbs = np.zeros((2, n_sim))
    # devide the total simulation time in dev sections
    dev = 1
    f_dbs = 130
    
    if (synaptic_fidelity != 0):
        dev = 3
    
    # for DBS on all the time
    if (dev == 1):
        dbs_duration = n_sim
        T_dbs = Fs/f_dbs
        dbs = np.arange(1, dbs_duration, np.round(T_dbs))

        I_dbs_full = np.zeros((1, dbs_duration))
        for i in dbs:
            I_dbs_full[0][int(i - 1)] = 0.02
        # I_dbs_full[dbs] = .02 # Amplitude of DBS
        I_dbs_pre = I_dbs_full
        T_dbs = Fs/f_dbs
    else:
        dbs_duration = (n_sim - chop_till)/dev # in seconds
        # multiplied by 10 to make the transmembrane voltage about 80 mv
        I_dbs_pre = dbsDelta(f_dbs, dbs_duration, dev, n_sim, Fs, chop_till)
        
    return [I_dbs, I_dbs_pre, dev]
    # return {'I_dbs': I_dbs, 'I_dbs_pre': I_dbs_pre, 'dev': dev}