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
import seaborn as sns
sns.set()

seed(1)
random_factor = random()

# =============================================================================
# PARAMETERS
# =============================================================================

ms = 100           # 1 second in miliseconds
dt = 0.1           # time step = 1ms
sim_time = 100
sim_steps = int(sim_time/dt)    # 1 second in miliseconds
time = np.arange(1, sim_steps)

# Izhikevich neuron model
vp = 30     # voltage peak
vr = -65    # voltage threshold
rs_params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}  # Regular Spiking

a_rs = rs_params['a']
b_rs = rs_params['b']
c_rs = rs_params['c'] + 15*random_factor**2
d_rs = rs_params['d'] - 6*random_factor**2

I_rs = 3.5*7

# Tsodkys and Markram synapse model
t_s_E = 3         # decay time constante of I (PSC current)
t_s_I = 11        # decay time constante of I (PSC current)

## Facilitation synapses parameters for excitatory synapses
t_f_E_F = 670
t_d_E_F = 138
U_E_F = 0.09
A_E_F = 0.8

## Facilitation synapses parameters for inhibitory synapses
t_f_I_F = 376
t_d_I_F = 45
U_I_F = 0.016
A_I_F = 0.2

## Depression synapses parameters for excitatory synapses
t_f_E_D = 17
t_d_E_D = 671
U_E_D = 0.5
A_E_D = 0.2

## Depression synapses parameters for inhibitory synapses
t_f_I_D = 21
t_d_I_D = 706
U_I_D = 0.25
A_I_D = 0.8

# =============================================================================
# VARIABLES
# =============================================================================
R_E_D = np.zeros((1, sim_steps)) # R for Excitatory Depression
u_E_D = np.zeros((1, sim_steps)) # u for Excitatory Depression
I_E_D = np.zeros((1, sim_steps)) # I for Excitatory Depression
R_E_D[0][0] = 1

R_E_F = np.zeros((1, sim_steps)) # R for Excitatory Facilitation
u_E_F = np.zeros((1, sim_steps)) # u for Excitatory Facilitation
I_E_F = np.zeros((1, sim_steps)) # I for Excitatory Facilitation
R_E_F[0][0] = 1

R_I_D = np.zeros((1, sim_steps)) # R for Inhibitory Depression
u_I_D = np.zeros((1, sim_steps)) # u for Inhibitory Depression
I_I_D = np.zeros((1, sim_steps)) # I for Inhibitory Depression
R_I_D[0][0] = 1

R_I_F = np.zeros((1, sim_steps)) # R for Inhibitory Facilitation
u_I_F = np.zeros((1, sim_steps)) # u for Inhibitory Facilitation
I_I_F = np.zeros((1, sim_steps)) # I for Inhibitory Facilitation
R_I_F[0][0] = 1

n_rs = 1  # number of RS neurons

v_rs = np.zeros((n_rs, sim_steps)); v_rs[0][0] = vr
u_rs = np.zeros((n_rs, sim_steps)); u_rs[0][0] = vr*rs_params['b']

PSC_E_D = np.zeros((n_rs, sim_steps))
PSC_E_F = np.zeros((n_rs, sim_steps))
PSC_I_D = np.zeros((n_rs, sim_steps))
PSC_I_F = np.zeros((n_rs, sim_steps))

# =============================================================================
# EQUATIONS IN FORM OF FUNCTIONS
# =============================================================================

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

def synapse_utilization(u, tau_f, U, AP, dt):
    return -(dt/tau_f)*u + U*(1 - u)*AP

def synapse_recovery(R, tau_d, u_next, AP, dt):
    return (dt/tau_d)*(1 - R) - u_next*R*AP

def synapse_current(I, tau_s, A, R, u_next, AP, dt):
    return -(dt/tau_s)*I + A*R*u_next*AP

# =============================================================================
# MAIN - STD
# =============================================================================
for t in time:
    # RS NEURON - Excitatory
    AP_aux = 0
    for i in range(n_rs):
        v_aux_rs = 1*v_rs[i][t - 1]
        u_aux_rs = 1*u_rs[i][t - 1]
        # Synapse var - Excitatory - Depression
        syn_u_aux_E_D = 1*u_E_D[i][t - 1]
        syn_R_aux_E_D = 1*R_E_D[i][t - 1]
        syn_I_aux_E_D = 1*I_E_D[i][t - 1]
        # Synapse var - Excitatory - Facilitation
        syn_u_aux_E_F = 1*u_E_F[i][t - 1]
        syn_R_aux_E_F = 1*R_E_F[i][t - 1]
        syn_I_aux_E_F = 1*I_E_F[i][t - 1]
        # Synapse var - Inhibitory - Depression
        syn_u_aux_I_D = 1*u_I_D[i][t - 1]
        syn_R_aux_I_D = 1*R_I_D[i][t - 1]
        syn_I_aux_I_D = 1*I_I_D[i][t - 1]
        # Synapse var - Inhibitory - Faciliation
        syn_u_aux_I_F = 1*u_I_F[i][t - 1]
        syn_R_aux_I_F = 1*R_I_F[i][t - 1]
        syn_I_aux_I_F = 1*I_I_F[i][t - 1]        
                
        # Neuron - FS - Excitatory
        if (v_aux_rs >= vp):
            AP_aux = 1
            v_aux_rs = v_rs[i][t]
            v_rs[i][t] = c_rs
            u_rs[i][t] = u_aux_rs + d_rs
        else:
            AP_aux = 0
            dv_rs = izhikevich_dvdt(v = v_aux_rs, u = u_aux_rs, I = I_rs)
            du_rs = izhikevich_dudt(v = v_aux_rs, u = u_aux_rs, a = a_rs, b = b_rs)
        
            v_rs[i][t] = v_aux_rs + dt*dv_rs
            u_rs[i][t] = u_aux_rs + dt*du_rs
            
        # Synapse - Excitatory - Depression
        syn_du_E_D = synapse_utilization(u = syn_u_aux_E_D, 
                                         tau_f = t_f_E_D, 
                                         U = U_E_D, 
                                         AP = AP_aux, 
                                         dt = dt)
        u_E_D[i][t] = syn_u_aux_E_D + syn_du_E_D
        
        syn_dR_E_D = synapse_recovery(R = syn_R_aux_E_D, 
                                  tau_d = t_d_E_D, 
                                  u_next = u_E_D[i][t], 
                                  AP = AP_aux, 
                                  dt = dt)
        R_E_D[i][t] = syn_R_aux_E_D + syn_dR_E_D
        
        syn_dI_E_D = synapse_current(I = syn_I_aux_E_D, 
                                 tau_s = t_s_E, 
                                 A = A_E_D, 
                                 R = syn_R_aux_E_D, 
                                 u_next = u_E_D[i][t], 
                                 AP = AP_aux, 
                                 dt = dt)
        I_E_D[i][t] = syn_I_aux_E_D + syn_dI_E_D
        PSC_E_D[i][t] = I_E_D[i][t]
        
        # Synapse - Excitatory - Facilitation
        syn_du_E_F = synapse_utilization(u = syn_u_aux_E_F, 
                                          tau_f = t_f_E_F, 
                                          U = U_E_F, 
                                          AP = AP_aux, 
                                          dt = dt)
        u_E_F[i][t] = syn_u_aux_E_F + syn_du_E_F
        
        syn_dR_E_F = synapse_recovery(R = syn_R_aux_E_F, 
                                      tau_d = t_d_E_F, 
                                      u_next = u_E_F[i][t], 
                                      AP = AP_aux, 
                                      dt = dt)
        R_E_F[i][t] = syn_R_aux_E_F + syn_dR_E_F
        
        syn_dI_E_F = synapse_current(I = syn_I_aux_E_F, 
                                      tau_s = t_s_E, 
                                      A = A_E_F, 
                                      R = syn_R_aux_E_F, 
                                      u_next = u_E_F[i][t], 
                                      AP = AP_aux, 
                                      dt = dt)
        I_E_F[i][t] = syn_I_aux_E_F + syn_dI_E_F
        PSC_E_F[i][t] = I_E_F[i][t]
        
        # Synapse - Inhibitory - Depression
        syn_du_I_D = synapse_utilization(u = syn_u_aux_I_D, 
                                          tau_f = t_f_I_D, 
                                          U = U_I_D, 
                                          AP=AP_aux, 
                                          dt = dt)
        u_I_D[i][t] = syn_u_aux_I_D + syn_du_I_D
        
        syn_dR_I_D = synapse_recovery(R = syn_R_aux_I_D, 
                                  tau_d = t_d_I_D, 
                                  u_next = u_I_D[i][t], 
                                  AP = AP_aux, 
                                  dt = dt)
        R_I_D[i][t] = syn_R_aux_I_D + syn_dR_I_D
        
        syn_dI_I_D = synapse_current(I = syn_I_aux_I_D, 
                                  tau_s = t_s_I, 
                                  A = A_I_D, 
                                  R = syn_R_aux_I_D, 
                                  u_next = u_I_D[i][t], 
                                  AP = AP_aux, 
                                  dt = dt)
        I_I_D[i][t] = syn_I_aux_I_D + syn_dI_I_D
        PSC_I_D[i][t] = I_I_D[i][t]
        
        # Synapse - Inhibitory - Facilitation
        syn_du_I_F = synapse_utilization(u = syn_u_aux_I_F, 
                                          tau_f = t_f_I_F, 
                                          U = U_I_F, 
                                          AP=AP_aux, 
                                          dt = dt)
        u_I_F[i][t] = syn_u_aux_I_F + syn_du_I_F
        
        syn_dR_I_F = synapse_recovery(R = syn_R_aux_I_F, 
                                      tau_d = t_d_I_F, 
                                      u_next = u_I_F[i][t], 
                                      AP = AP_aux, 
                                      dt = dt)
        R_I_F[i][t] = syn_R_aux_I_F + syn_dR_I_F
        
        syn_dI_I_F = synapse_current(I = syn_I_aux_I_F, 
                                      tau_s = t_s_I, 
                                      A = A_I_F, 
                                      R = syn_R_aux_I_F, 
                                      u_next = u_I_F[i][t], 
                                      AP = AP_aux, 
                                      dt = dt)
        I_I_F[i][t] = syn_I_aux_I_F + syn_dI_I_F
        PSC_I_F[i][t] = I_I_F[i][t]


print_comparison(voltage = v_rs[0], PSC = PSC_E_D[0], title = 'Regular Spiking - Excitatory - Depression')
print_comparison(voltage = v_rs[0], PSC = PSC_E_F[0], title = 'Regular Spiking - Excitatory - Facilitation')
print_comparison(voltage = v_rs[0], PSC = PSC_I_D[0], title = 'Regular Spiking - Inhibitory - Depression')
print_comparison(voltage = v_rs[0], PSC = PSC_I_F[0], title = 'Regular Spiking - Inhibitory - Facilitation')
