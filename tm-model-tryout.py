"""
Created on Wed Nov 15 21:27:23 2023

Depression
- RS to FS

Facilitation
- RS to LTS


@author: celinesoeiro
"""
import matplotlib.pyplot as plt
import numpy as np

from izhikevich_neuron import izhikevich_neuron

# =============================================================================
# PARAMETERS
# =============================================================================

ms = 1000           # 1 second in miliseconds
dt = 1/ms           # time step = 1ms
sim_time = 1*ms     # 1 second in miliseconds

# Izhikevich neuron model
vp = 30     # voltage peak
vr = -65    # voltage threshold
cv = 4      # current value
ex_neuron_params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}, # Regular Spiking
in_neuron_params = {'a': 0.1, 'b': 0.2, 'c': -65, 'd': 2},  # Fast Spiking


# Tsodkys and Markram synapse model
## With pseudo-linear synapse contribution
## Excitatory synapses (column 0 is for STF synapses and column 1 is for STD synapses)
tau_f_E_p = [670, 17, 326]      # decay time constant of u (resources ready for use)
tau_d_E_p = [138, 671, 329]     # recovery time constant of x (available resources)
tau_s_E_p = 3                   # decay time constante of I (PSC current)
U_E_p = [0.09, 0.5, 0.29]       # increment of u produced by a spike
A_E_p = [0.2, 0.63, 0.17]       # absolute synaptic efficacy

## Inhibitory synapses (column 0 is for STF synapses and column 1 is for STD synapses)
tau_f_I_p = [376, 21, 62]       # decay time constant of u (resources ready for use)
tau_d_I_p = [45, 706, 144]      # recovery time constant of x (available resources)
tau_s_I_p = 11                  # decay time constante of I (PSC current)
U_I_p = [0.016, 0.25, 0.32]     # increment of u produced by a spike
A_I_p = [0.08, 0.75, 0.17]      # absolute synaptic efficacy

## Without pseudo-linear synapse contribution
## Excitatory synapses (column 0 is for STF synapses and column 1 is for STD synapses)
tau_f_E = [670, 17]     # decay time constant of u (resources ready for use)
tau_d_E = [138, 671]    # recovery time constant of x (available resources)
tau_s_E = 3             # decay time constante of I (PSC current)
U_E = [0.09, 0.5]       # increment of u produced by a spike
A_E = [0.37, 0.63]      # absolute synaptic efficacy

## Inhibitory synapses (column 0 is for STF synapses and column 1 is for STD synapses)
tau_f_I = [376, 21]     # decay time constant of u (resources ready for use)
tau_d_I = [45, 706]     # recovery time constant of x (available resources)
tau_s_I = 11            # decay time constante of I (PSC current)
U_I = [0.016, 0.25]     # increment of u produced by a spike
A_I = [0.08, 0.92]      # absolute synaptic efficacy

# =============================================================================
# VARIABLES
# =============================================================================
u_E = np.zeros((1, 3))  # 
x_E = np.zeros((1, 3))
Is_E = np.zeros((1, 3))

u_E_p = np.zeros((1, 3))  # 
x_E_p = np.zeros((1, 3))
Is_E_p = np.zeros((1, 3))

u_I = np.zeros((1, 3))  # 
x_I = np.zeros((1, 3))
Is_I = np.zeros((1, 3))

u_I_p = np.zeros((1, 3))  # 
x_I_p = np.zeros((1, 3))
Is_I_p = np.zeros((1, 3))

# =============================================================================
# EQUATIONS IN FORM OF FUNCTIONS
# =============================================================================
def tm_synapse_eq(u, x, Is, AP, tau_f, tau_d, tau_s, U, A, dt, pseudo_linear = False):
    length = 2
    if(pseudo_linear):
        length = 3        
    # Solve EDOs using Euler method
    for p in range(length):
        # variable just after the spike
        next_u = u[0][p] + U[p]*(1 - u[0][p]) 
        # u -> utilization factor -> resources ready for use
        u[0][p] = u[0][p] + dt*(-u[0][p]/tau_f[p] + U[p]*(1 - u[0][p])*AP)
        # x -> availabe resources -> Fraction of resources that remain available after neurotransmitter depletion
        x[0][p] = x[0][p] + dt*((1 - x[0][p])/tau_d[p] - next_u*x[0][p]*AP)
        # PSC
        Is[0][p] = Is[0][p] + dt*(-Is[0][p]/tau_s + A[p]*x[0][p]*next_u*AP)
        
    Ipost = np.sum(Is)
    
    tm_syn_inst = dict()
    tm_syn_inst['u'] = u
    tm_syn_inst['x'] = x
    tm_syn_inst['Is'] = Is
    tm_syn_inst['Ipost'] = np.around(Ipost, decimals=6)
        
    return tm_syn_inst

def get_PSC(voltage, vp, u, x, Is, tau_f, tau_d, tau_s, U, A, dt):
    tm_values = []
    for i in voltage:
        if (i > vp):
            tm = tm_synapse_eq(u, x, Is, 1, tau_f, tau_d, tau_s, U, A, dt)
            tm_values.append(tm['Ipost'])
        else:
            tm = tm_synapse_eq(u, x, Is, 0, tau_f, tau_d, tau_s, U, A, dt)
            tm_values.append(tm['Ipost'])
    
    return tm_values

def print_signal(signal, title):
    plt.figure(1)
    plt.suptitle(title)
    plt.plot(signal, 'b')
    plt.grid(True)
    plt.show()

def print_comparison(voltage, PSC, title):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 11))
    fig.suptitle(title)
    ax1.set_title('Voltage')
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    ax1.xaxis.grid(True, linestyle='-', which='major', color='lightcoral',
                   alpha=0.5)
    ax2.set_title('PSC')
    ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    ax2.xaxis.grid(True, linestyle='-', which='major', color='lightcoral',
                   alpha=0.5)

    ax1.plot(voltage)
    ax2.plot(PSC)

# =============================================================================
# MAIN
# =============================================================================
# creating the neuron - Excitatory
v_RS, I_RS = izhikevich_neuron(
    params = ex_neuron_params[0],
    current_value = cv,
    current_start = 0,    
    current_finish = -1,    
    voltage_pick = vp,
    simulation_time = sim_time, 
    time_step = dt,
    neuron_type = 'excitatory',
    initial_voltage = vr
    )

# ploting the neuron membrane voltage
print_signal(v_RS, "Izhikevich model - Regular Spiking - Excitatory connection")

# Finding the PSC of the created neuron
tm_values_E = get_PSC(v_RS, vp, u_E, x_E, Is_E, tau_f_E, tau_d_E, tau_s_E, U_E, A_E, dt)

# Finding the PSC of the created neuron - Pseudo Linear
tm_values_E_p = get_PSC(v_RS, vp, u_E_p, x_E_p, Is_E_p, tau_f_E_p, tau_d_E_p, tau_s_E_p, U_E_p, A_E_p, dt)

# Plotting the PSC
print_signal(tm_values_E, "TM model - STF - Regular Spiking Post Synaptic Response")

# Plotting the PSC
print_signal(tm_values_E_p, "TM model - STF - Regular Spiking Post Synaptic Response - With PL")

# =============================================================================

# creating the neuron - Inhibitory
v_FS, I_FS = izhikevich_neuron(
    params = in_neuron_params[0],
    current_value = cv,
    current_start = 0,
    current_finish = -1,
    voltage_pick = vp,
    simulation_time = sim_time,
    time_step = dt,
    neuron_type = 'inhibitory',
    )

# ploting the neuron membrane voltage
print_signal(v_FS, "Izhikevich model - Fast Spiking - Inhibitory connection - With PL")

# Finding the PSC of the created neuron
tm_values_I = get_PSC(v_FS, vp, u_I, x_I, Is_I, tau_f_I, tau_d_I, tau_s_I, U_I, A_I, dt)

# Finding the PSC of the created neuron
tm_values_I_p = get_PSC(v_FS, vp, u_I_p, x_I_p, Is_I_p, tau_f_I_p, tau_d_I_p, tau_s_I_p, U_I_p, A_I_p, dt)

# ploting the PSC
print_signal(tm_values_I, "TM model - STD - Regular Spiking Post Synaptic Response")

# ploting the PSC
print_signal(tm_values_I_p, "TM model - STD - Regular Spiking Post Synaptic Response - With PL")

# =============================================================================
# Creating better plots
# =============================================================================

print_comparison(v_RS, tm_values_E, 'STF-dominated synapse of an excitatory neuron')

print_comparison(v_FS, tm_values_I, 'STD-dominated synapse of an inhibitory neuron')

print_comparison(v_RS, tm_values_E_p, 'STF-dominated synapse of an excitatory neuron - With PL')

print_comparison(v_FS, tm_values_I_p, 'STD-dominated synapse of an inhibitory neuron - With PL')

# =============================================================================
# Comparing the signals with and without pseudo-linear
# =============================================================================
# Calculate the magnitude or amplitude of each signal
magnitude_PSC_E = np.abs(tm_values_E)
magnitude_PSC_E_p = np.abs(tm_values_E_p)

# Find common indices where values are nonzero
common_indices_E = np.intersect1d(np.nonzero(magnitude_PSC_E)[0], np.nonzero(magnitude_PSC_E_p)[0])

magnitude_E_nonzero = magnitude_PSC_E[common_indices_E]
magnitude_E_p_nonzero = magnitude_PSC_E_p[common_indices_E]

# Calculate the gain as the ratio of non-zero magnitudes
gain_E = magnitude_E_p_nonzero/magnitude_E_nonzero

# Calculate the Signal-to-Noise Ratio (SNR) in decibels (dB) for non-zero values
snr_E_db = 10*np.log10(gain_E**2)

print("Gain: ", gain_E)
print("SNR (db): ", snr_E_db)
print("average gain: ", np.mean(gain_E))

magnitude_PSC_I = np.abs(tm_values_I)
magnitude_PSC_I_p = np.abs(tm_values_I_p)

# Find common indices where values are nonzero
common_indices_I = np.intersect1d(np.nonzero(magnitude_PSC_I)[0], np.nonzero(magnitude_PSC_I_p)[0])

magnitude_I_nonzero = magnitude_PSC_I[common_indices_I]
magnitude_I_p_nonzero = magnitude_PSC_I_p[common_indices_I]

# Calculate the gain as the ratio of non-zero magnitudes
gain_I = magnitude_I_p_nonzero/magnitude_I_nonzero

# Calculate the Signal-to-Noise Ratio (SNR) in decibels (dB) for non-zero values
snr_dI_b = 10*np.log10(gain_I**2)

print("Gain: ", gain_I)
print("SNR (db): ", snr_dI_b)
print("average gain: ", np.mean(gain_I))