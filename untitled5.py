import numpy as np
import matplotlib.pyplot as plt

# Izhikevich neuron model parameters
a = 0.02
b = 0.2
c = -65.0
d = 8.0

# Simulation parameters
sim_time = 3.0       # Simulation time (seconds)
dt = 0.01            # Time step (seconds), equivalent to 10 ms

def poisson_spike_generator(sim_time, dt, num_neurons, thalamic_firing_rate):
    # Calculate the number of time steps
    num_steps = int(sim_time / dt)

    # Initialize an array to store spike times for each neuron
    spike_times = [[] for _ in range(num_neurons)]

    # Calculate firing probability
    firing_prob = thalamic_firing_rate * dt  # Calculate firing probability

    # Generate spikes for each neuron using the Poisson distribution
    for t in range(num_steps):
        for neuron_id in range(num_neurons):
            # Generate a random number between 0 and 1
            rand_num = np.random.rand()
            
            # If the random number is less than the firing probability, spike
            if rand_num < firing_prob:
                spike_times[neuron_id].append(t * dt)

    return spike_times


# # Define the Izhikevich neuron model function
# def izhikevich_neuron(input_current):
#     num_steps = len(input_current)
    
#     v = np.zeros(num_steps)
#     u = np.zeros(num_steps)
#     v[0] = -65.0  # Initial membrane potential (mV)
#     u[0] = b * v[0]  # Initial recovery variable
    
#     spike_times = []  # Store spike times
    
#     for t in range(1, num_steps):
#         # Update membrane potential and recovery variable
#         dv = (0.04 * v[t - 1] ** 2) + (5 * v[t - 1]) + 140 - u[t - 1] + input_current[t]
#         du = a * (b * v[t - 1] - u[t - 1])
        
#         v[t] = v[t - 1] + (dv * dt)
#         u[t] = u[t - 1] + (du * dt)
        
#         # Check for a spike and reset if the threshold is crossed
#         if v[t] >= 30:
#             v[t] = c
#             u[t] += d
#             spike_times.append(t * dt)
    
#     return v, spike_times

# # Generate spikes using the Poisson spike generator
# num_neurons = 1
# thalamic_firing_rate = 20.0  # Average firing rate in Hz
# spike_times_generator = poisson_spike_generator(sim_time, dt, num_neurons, thalamic_firing_rate)

# # Convert spike times to an input current
# input_current = np.zeros(int(sim_time / dt))
# for spike_times in spike_times_generator:
#     spike_indices = np.array(spike_times) / dt
#     input_current[spike_indices.astype(int)] = 250.0  # Adjust input current as needed

# # Simulate the Izhikevich neuron with the input current
# membrane_potential, spike_times_neuron = izhikevich_neuron(input_current)

# # Plot the membrane potential and spike times of the neuron
# time_axis = np.arange(0, sim_time, dt)
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# plt.plot(time_axis, membrane_potential)
# plt.xlabel('Time (s)')
# plt.ylabel('Membrane Potential (mV)')
# plt.title('Izhikevich Neuron Model')

# plt.subplot(2, 1, 2)
# plt.scatter(spike_times_neuron, [0] * len(spike_times_neuron), c='r', marker='|', linewidths=0.5)
# plt.xlabel('Time (s)')
# plt.ylabel('Spike')
# plt.title('Neuron Spike Times')
# plt.tight_layout()
# plt.show()

# =============================================================================
# 
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# Izhikevich neuron model parameters
a = 0.02
b = 0.2
c = -65.0
d = 8.0

# Tsodyks-Markram synaptic model parameters
U = 0.5  # Utilization of synaptic efficacy
tau_rec = 100.0  # Recovery time constant (ms)
tau_facil = 100.0  # Facilitation time constant (ms)

# Simulation parameters
sim_time = 3.0       # Simulation time (seconds)
dt = 0.01            # Time step (seconds), equivalent to 10 ms

# Define the Izhikevich neuron model function with Tsodyks-Markram synapses
def izhikevich_neuron(input_current, layer_name):
    num_neurons = len(input_current)
    num_steps = len(input_current[0])
    
    v = np.zeros((num_neurons, num_steps))
    u = np.zeros((num_neurons, num_steps))
    v[:, 0] = -70.0  # Initial membrane potential (mV)
    u[:, 0] = b * v[:, 0]  # Initial recovery variable
    
    spike_times = [[] for _ in range(num_neurons)]  # Store spike times
    
    # Initialize synaptic variables
    x = np.zeros((num_neurons, num_steps))
    u_synaptic = np.ones((num_neurons, num_steps)) * U
    
    for t in range(1, num_steps):
        for neuron_id in range(num_neurons):
            # Update synaptic variables using Tsodyks-Markram model for each neuron
            x[neuron_id, t] = x[neuron_id, t - 1] + (dt / tau_rec) * (1 - x[neuron_id, t - 1]) - (dt / tau_facil) * u_synaptic[neuron_id, t - 1] * x[neuron_id, t - 1]
            u_synaptic[neuron_id, t] = u_synaptic[neuron_id, t - 1] + (dt / tau_rec) * (U - u_synaptic[neuron_id, t - 1]) * x[neuron_id, t - 1]
            
            # Update membrane potential and recovery variable for each neuron
            dv = (0.04 * v[neuron_id, t - 1] ** 2) + (5 * v[neuron_id, t - 1]) + 140 - u[neuron_id, t - 1] + input_current[neuron_id][t]
            du = a * (b * v[neuron_id, t - 1] - u[neuron_id, t - 1])
            
            v[neuron_id, t] = v[neuron_id, t - 1] + (dv * dt)
            u[neuron_id, t] = u[neuron_id, t - 1] + (du * dt)
            
            # Check for spikes and reset if the threshold is crossed for each neuron
            if v[neuron_id, t] >= 30:
                v[neuron_id, t] = c
                u[neuron_id, t] += d
                spike_times[neuron_id].append(t * dt)
                
                # Update synaptic variables for the post-synaptic neuron
                x[neuron_id, t] += u_synaptic[neuron_id, t]
                u_synaptic[neuron_id, t] *= U
    
    return v, spike_times

# Function to create thalamic input for specified layers
def generate_thalamic_input(sim_time, dt, num_neurons, thalamic_firing_rate, layers):
    spike_times_generator = poisson_spike_generator(sim_time, dt, num_neurons, thalamic_firing_rate)
    
    # Convert spike times to input current for specified layers
    input_current = [[] for _ in range(len(layers))]
    for i, layer in enumerate(layers):
        for _ in range(num_neurons):
            input_current[i].append(np.zeros(int(sim_time / dt)))
        
        for spike_times in spike_times_generator:
            spike_indices = np.array(spike_times) / dt
            for neuron_id in range(num_neurons):
                input_current[i][neuron_id][spike_indices.astype(int)] = 10.0  # Adjust input current as needed
    
    return input_current

# Generate spikes using the Poisson spike generator
num_neurons_per_layer = 100  # Number of neurons per layer
thalamic_firing_rate = 20.0  # Average firing rate in Hz

layers = ['Layer D', 'Cortical Interneurons']  # Specify layers that receive thalamic input
input_current = generate_thalamic_input(sim_time, dt, num_neurons_per_layer, thalamic_firing_rate, layers)

# Simulate neurons in Layer D
membrane_potential_d, spike_times_d = izhikevich_neuron(input_current[0], layers[0])

# Simulate cortical interneurons
num_interneurons = num_neurons_per_layer
membrane_potential_interneurons, spike_times_interneurons = izhikevich_neuron(input_current[1], layers[1])

# Plot the membrane potential and spike times of neurons in Layer D
# Plot the membrane potential and spike times of the neuron
time_axis = np.arange(0, sim_time, dt)
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time_axis, membrane_potential_interneurons)
plt.xlabel('Time (s)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Izhikevich Neuron Model')

plt.subplot(2, 1, 2)
plt.scatter(spike_times_interneurons, [0] * len(spike_times_interneurons), c='r', marker='|', linewidths=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Spike')
plt.title('Neuron Spike Times')
plt.tight_layout()
plt.show()
