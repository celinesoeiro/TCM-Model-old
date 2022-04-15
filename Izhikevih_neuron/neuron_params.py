# -*- coding: utf-8 -*-
"""
@author: Celine Soeiro

@description: Popular neuron parameters

**_FL -> Parameters obtained from the Farokhniaee and Lowery model

"""

def regular_spiking():
    params = {
        'params': {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}, 
        'current_value': 10,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'excitatory',
        'initial_voltage': -65
    }
    return params

def intrinsically_bursting():
    params = {
        'params': {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 4}, 
        'current_value': 10,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'excitatory',
    }
    return params

def chattering():
    params = {
        'params': {'a': 0.02, 'b': 0.2, 'c': -50, 'd': 2}, 
        'current_value': 10,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'excitatory',
    }
    return params

def fast_spiking():
    params = {
        'params': {'a': 0.1, 'b': 0.2, 'c': -65, 'd': 2},
        'current_value': 10,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'inhibitory',
    }
    return params

def thalamo_cortical_depolarized():
    params = {
        'params': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 0.05},
        'current_value': 10,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'excitatory',
        'initial_voltage': -60
    }
    return params

def thalamo_cortical_hyperpolarized():
    params = {
        'params': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 0.05}, 
        'current_value': 10,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'excitatory',
        'initial_voltage': -90
    }
    return params

def resonator():
    params = {
        'params': {'a': 0.1, 'b': 0.26, 'c': -65, 'd': -1}, 
        'current_value': 10,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'excitatory'
    }
    return params

def low_threshold_spiking():
    params = {
        'params': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2},
        'current_value': 10,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'inhibitory'
    }
    return params

# =============================================================================
# Advanced Models
# =============================================================================

def tonic_spiking():
    params = {
        'params': {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 6},
        'current_value': 14,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'excitatory',
        'initial_voltage': -70,
    }
    return params

def phasic_spiking():
    params = {
        'params': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 6},
        'current_value': 0.5,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'excitatory',
        'initial_voltage': -64,
    }
    return params

def tonic_bursting():
    params = {
        'params': {'a': 0.02, 'b': 0.2, 'c': -50, 'd': 2},
        'current_value': 15,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'excitatory',
        'initial_voltage': -70,
    }
    return params

def phasic_bursting():
    params = {
        'params': {'a': 0.02, 'b': 0.25, 'c': -55, 'd': 0.05},
        'current_value': 0.6,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'excitatory',
        'initial_voltage': -64,
    }
    return params

def mixed_mode():
    params = {
        'params': {'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4},
        'current_value': 10,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'excitatory',
        'initial_voltage': -70,
    }
    return params

def spike_frequency_adapt():
    params = {
        'params': {'a': 0.01, 'b': 0.2, 'c': -65, 'd': 8},
        'current_value': 30,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'excitatory',
        'initial_voltage': -70,
    }
    return params

# =============================================================================
# ================================= FL MODEL
# =============================================================================

def regular_spiking_FL():
    params = {
        'params': {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8},
        'current_value': 2.5,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'excitatory',
        'initial_voltage': -65
    }
    return params

def intrinsically_bursting_FL():
    params = {
        'params': {'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4},
        'current_value': 2.5,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'excitatory',
    }
    return params

def fast_spiking_FL():
    params = {
        'params': {'a': 0.1, 'b': 0.2, 'c': -65, 'd': 2},
        'current_value': 3.2,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'inhibitory',
    }
    return params

def low_thresholding_spiking_FL():
    params = {
        'params': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2},
        'current_value': 0,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'inhibitory',
    }
    return params

def thalamo_cortical_FL():
    params = {
        'params': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 0.05},
        'current_value': 0,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'excitatory',
    }
    return params

def thalamo_reticular_FL():
    params = {
        'params': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2.05},
        'current_value': 0,
        'current_start': 100,
        'current_finish':  900,
        'voltage_pick': 30,
        'simulation_time': 100,
        'time_step': 0.1,
        'neuron_type': 'inhibitory',
    }
    return params