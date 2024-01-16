"""
Created on Wed Nov 15 21:27:23 2023

Depression
- RS to FS

Facilitation
- RS to LTS

STF - Short Term Facilitation
STD - Short Term Depression
PL - Pseudo Linear

R - Recovered state - synaptic efficacy
u - Utilization state - synaptic efficacy

@author: celinesoeiro
"""
import matplotlib.pyplot as plt
import numpy as np
from random import seed, random

seed(1)
random_factor = random()

# =============================================================================
# PARAMETERS
# =============================================================================

ms = 1000           # 1 second in miliseconds
dt = 1/ms           # time step = 1ms
sim_time = 2*ms    # 1 second in miliseconds
time = np.arange(1, sim_time + 1, 1)

# Izhikevich neuron model
vp = 30     # voltage peak
vr = -65    # voltage threshold
rs_params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}  # Regular Spiking
fs_params = {'a': 0.1, 'b': 0.2, 'c': -65, 'd': 2}   # Fast Spiking
lts_params = {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2} # Low Threshold Spiking

I_rs = 3.6
I_fs = 3.7
I_lts = 3.9

# Tsodkys and Markram synapse model

## With pseudo-linear synapse contribution
## column 0 is for STF synapses, column 1 is for STD synapses and column 2 is for PL synapses

p = 3 # if pseudo-linear is considered

t_f_E = [670, 17, 326]      # decay time constant of u (resources ready for use)
t_d_E = [138, 671, 329]     # recovery time constant of x (available resources)
t_s_E = 3                   # decay time constante of I (PSC current)
U_E_p = [0.09, 0.5, 0.29]       # increment of u produced by a spike
A_E_p = [0.2, 0.63, 0.17]       # absolute synaptic efficacy

t_f_I = [376, 21, 62]       # decay time constant of u (resources ready for use)
t_d_I = [45, 706, 144]      # recovery time constant of x (available resources)
t_s_I = 11                  # decay time constante of I (PSC current)
U_I_p = [0.016, 0.25, 0.32]     # increment of u produced by a spike
A_I_p = [0.08, 0.75, 0.17]      # absolute synaptic efficacy

## Without pseudo-linear synapse contribution
## column 0 is for STF synapses and column 1 is for STD synapses

p = 2 # if pseudo-linear is NOT considered

tau_f_E = [670, 17]     # decay time constant of u (resources ready for use)
tau_d_E = [138, 671]    # recovery time constant of x (available resources)
tau_s_E = 3             # decay time constante of I (PSC current)
U_E = [0.09, 0.5]       # increment of u produced by a spike
A_E = [0.37, 0.63]      # absolute synaptic efficacy

tau_f_I = [376, 21]     # decay time constant of u (resources ready for use)
tau_d_I = [45, 706]     # recovery time constant of x (available resources)
tau_s_I = 11            # decay time constante of I (PSC current)
U_I = [0.016, 0.25]     # increment of u produced by a spike
A_I = [0.08, 0.92]      # absolute synaptic efficacy

# =============================================================================
# VARIABLES
# =============================================================================
u_E = np.zeros((1, 3)) 
R_E = np.zeros((1, 3))
I_E = np.zeros((1, 3))

u_I = np.zeros((1, 3))  
R_I = np.zeros((1, 3))
I_I = np.zeros((1, 3))

n_rs = 1  # number of RS neurons
n_fs = 1  # number of FS neurons
n_lts = 1 # number of LTS neurons

v_rs = np.zeros((n_rs, len(time))); v_rs[0][0] = vr
u_rs = np.zeros((n_rs, len(time))); u_rs[0][0] = vr*rs_params['b']

v_fs = np.zeros((n_fs, len(time))); v_fs[0][0] = vr
u_fs = np.zeros((n_fs, len(time))); u_fs[0][0] = vr*fs_params['b']

v_lts = np.zeros((n_lts, len(time))); v_lts[0][0] = vr
u_lts = np.zeros((n_lts, len(time))); u_lts[0][0] = vr*lts_params['b']

# =============================================================================
# EQUATIONS IN FORM OF FUNCTIONS
# =============================================================================
def tm_synapse_eq(u, R, I, AP, t_f, t_d, t_s, U, A, dt):
    # Solve EDOs using Euler method
    for j in range(p):
        # variable just after the spike
        next_u = u[0][j] + U[j]*(1 - u[0][j]) 
        # u -> utilization factor -> resources ready for use
        u[0][j] = u[0][j] + dt*(-u[0][j]/t_f[j] + U[j]*(1 - u[0][j])*AP)
        # x -> availabe resources -> Fraction of resources that remain available after neurotransmitter depletion
        R[0][j] = R[0][j] + dt*((1 - R[0][j])/t_d[j] - next_u*R[0][j]*AP)
        # PSC
        I[0][j] = I[0][j] + dt*(-I[0][j]/t_s + A[p]*R[0][j]*next_u*AP)
        
    Ipost = np.sum(I)
    
    tm_syn_inst = dict()
    tm_syn_inst['u'] = u
    tm_syn_inst['R'] = R
    tm_syn_inst['I'] = I
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
    
def izhikevich_dvdt(v, u, I):
    return 0.04*v**2 + 5*v + 140 - u + I

def izhikevich_dudt(v, u, a, b):
    return a*(b*v - u)

# =============================================================================
# MAIN
# =============================================================================
for t in time:
    # RS NEURON
    for i in range(n_rs):
        AP_aux = 0
        v_aux = 1*v_rs[i][t - 1]
        u_aux = 1*u_rs[i][t - 1]
        
        a = rs_params['a']
        b = rs_params['b']
        c = rs_params['c'] + 15*random_factor**2
        d = rs_params['d'] - 6*random_factor**2
        
        if (v_aux >= vp):
            AP_aux = 1
            v_aux = v_rs[i][t]
            v_rs[i][t] = c
            u_rs[i][t] = u_aux + d
        else:
            dv = izhikevich_dvdt(v = v_aux, u = u_aux, I = I_rs)
            du = izhikevich_dudt(v = v_aux, u = u_aux, a = a, b = b)
        
            v_rs[i][t] = v_aux + dt*dv
            u_rs[i][t] = u_aux + dt*du
        

    # FS NEURON
    for i in range(n_rs):
        AP_aux = 0
        v_aux = 1*v_fs[i][t - 1]
        u_aux = 1*u_fs[i][t - 1]
        
        a = fs_params['a'] + 0.08*random_factor
        b = fs_params['b'] - 0.05*random_factor
        c = fs_params['c']
        d = fs_params['d']
        
        if (v_aux >= vp):
            AP_aux = 1
            v_aux = v_fs[i][t]
            v_fs[i][t] = c
            u_fs[i][t] = u_aux + d
        else:
            dv = izhikevich_dvdt(v = v_aux, u = u_aux, I = I_fs)
            du = izhikevich_dudt(v = v_aux, u = u_aux, a = a, b = b)
        
            v_fs[i][t] = v_aux + dt*dv
            u_fs[i][t] = u_aux + dt*du
            

print_signal(v_rs[0], "Izhikevich model - Regular Spiking")
print_signal(v_fs[0], "Izhikevich model - Fast Spiking")

