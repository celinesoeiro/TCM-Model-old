# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 19:12:58 2022

@author: Avell
"""

def izhikevich_neuron_instaneous(
        params: dict,
        neuron_type: str,
        voltage_pick: float,
        time_step: float,
        current_value: int,
        random_factor: float,
        initial_voltage = -65,
        ):
    
    # Check if paramaters exists, if dont display error msg
    if (not params.get('a') 
        or not params.get('b') 
        or not params.get('c') 
        or not params.get('d')
        ): 
        return 'Parameters must be a, b, c and d' 
    
    
    if (neuron_type == 'excitatory' or 'excit'):
        a = params['a']
        b = params['b']
        c = params['c'] + 15*random_factor**2
        d = params['d'] - 6*random_factor**2
    elif (neuron_type == 'inhibitory' or 'inhib'):
        a = params['a'] + 0.08*random_factor
        b = params['b'] - 0.05*random_factor
        c = params['c']
        d = params['d']
    else:
        return 'Neuron type must be excitatory or inhibitory'
    
    # Current input
    I = current_value
    
    # membrane potential 
    v = initial_voltage
    
    # membrane recovery variable
    u = b*v  
    
    # Izhikevich neuron equations
    def dvdt(v, u, I):
        return 0.04*v**2 + 5*v + 140 - u + I
    
    def dudt(v,u):
        return a*(b*v - u)
    
    dv = dvdt(v, u, I)
    du = dudt(v, u)
    v = v + dv*time_step
    u = u + du*time_step

    # return membrane potential and input current
    return v, u, c, d