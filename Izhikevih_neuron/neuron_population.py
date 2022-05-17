"""
@author: Celine Soeiro

@description: Neuron population of Izhikevich neurons

"""

from .izhikevich_neuron import izhikevich_neuron

def neuron_population(size: int, neuron_params: dict):
    if (size < 1):
        return 'Size must be grater than 1'
    
    # define the output, the voltage of each neuron in an array
    voltages = []
    currents = []
    
    # get the neuron params 
    params = neuron_params['params']
    current_value = neuron_params['current_value']
    current_start = neuron_params['current_start']
    current_finish = neuron_params['current_finish']
    voltage_pick = neuron_params['voltage_pick']
    simulation_time = neuron_params['simulation_time']
    time_step = neuron_params['time_step']
    neuron_type = neuron_params['neuron_type']
    initial_voltage = neuron_params['initial_voltage']
    
    # create the desired quantity of neurons
    for i in range(size):
        # initialize the neuron with the params
        neuron_voltage, neuron_current = izhikevich_neuron(
            params = params,
            current_value = current_value,
            current_start = current_start,
            current_finish = current_finish,
            voltage_pick = voltage_pick,
            simulation_time = simulation_time,
            time_step = time_step,
            neuron_type = neuron_type,
            initial_voltage = initial_voltage
        )

        voltages.insert(i,neuron_voltage)
        currents.insert(i,neuron_current)
    
    return voltages, currents