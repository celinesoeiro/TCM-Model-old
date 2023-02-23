# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 19:12:58 2022

@author: Celine
"""

def izhikevich_neuron_instaneous(
        params: dict,
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
    
    a = params['a']
    b = params['b']
    c = params['c'] 
    d = params['d'] 

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