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
dt = 1/ms           # time step = 1ms
sim_time = 100
sim_steps = int(sim_time/dt)    # 1 second in miliseconds
time = np.arange(1, sim_steps)

# Izhikevich neuron model
vp = 30     # voltage peak
vr = -65    # voltage threshold
rs_params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}  # Regular Spiking
fs_params = {'a': 0.1, 'b': 0.2, 'c': -65, 'd': 2}   # Fast Spiking
lts_params = {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2} # Low Threshold Spiking

I_rs = 3.5*7
I_fs = 3.8*4
I_lts = 0.6*7

# Tsodkys and Markram synapse model

## With pseudo-linear synapse contribution
## column 0 is for STF synapses, column 1 is for STD synapses and column 2 is for PL synapses

p = 3 # if pseudo-linear is considered

t_f_E = [670, 17, 326]      # decay time constant of u (resources ready for use)
t_d_E = [138, 671, 329]     # recovery time constant of x (available resources)
t_s_E = 3                   # decay time constante of I (PSC current)
U_E = [0.09, 0.5, 0.29]       # increment of u produced by a spike
A_E = [0.2, 0.63, 0.17]       # absolute synaptic efficacy

t_f_I = [376, 21, 62]       # decay time constant of u (resources ready for use)
t_d_I = [45, 706, 144]      # recovery time constant of x (available resources)
t_s_I = 11                  # decay time constante of I (PSC current)
U_I = [0.016, 0.25, 0.32]     # increment of u produced by a spike
A_I = [0.08, 0.75, 0.17]      # absolute synaptic efficacy

## Without pseudo-linear synapse contribution
## column 0 is for STF synapses and column 1 is for STD synapses

# p = 2 # if pseudo-linear is NOT considered

# t_f_E = [670, 17]     # decay time constant of u (resources ready for use)
# t_d_E = [138, 671]    # recovery time constant of x (available resources)
# t_s_E = 3             # decay time constante of I (PSC current)
# U_E = [0.09, 0.5]       # increment of u produced by a spike
# A_E = [0.37, 0.63]      # absolute synaptic efficacy

# t_f_I = [376, 21]     # decay time constant of u (resources ready for use)
# t_d_I = [45, 706]     # recovery time constant of x (available resources)
# t_s_I = 11            # decay time constante of I (PSC current)
# U_I = [0.016, 0.25]     # increment of u produced by a spike
# A_I = [0.08, 0.92]      # absolute synaptic efficacy

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

v_rs = np.zeros((n_rs, sim_steps)); v_rs[0][0] = vr
u_rs = np.zeros((n_rs, sim_steps)); u_rs[0][0] = vr*rs_params['b']

v_fs = np.zeros((n_fs, sim_steps)); v_fs[0][0] = vr
u_fs = np.zeros((n_fs, sim_steps)); u_fs[0][0] = vr*fs_params['b']

v_lts = np.zeros((n_lts, sim_steps)); v_lts[0][0] = vr
u_lts = np.zeros((n_lts, sim_steps)); u_lts[0][0] = vr*lts_params['b']

rs_syn = np.zeros((n_rs, sim_steps))
fs_syn = np.zeros((n_fs, sim_steps))
lts_syn = np.zeros((n_lts, sim_steps))

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
        I[0][j] = I[0][j] + dt*(-I[0][j]/t_s + A[j]*R[0][j]*next_u*AP)
        
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

PSC_rs = np.zeros((n_rs, sim_steps))

# =============================================================================
# MAIN - STD
# =============================================================================
for t in time:
    # RS NEURON - Excitatory
    AP_aux_rs = 0
    for i in range(n_rs):
        v_aux_rs = 1*v_rs[i][t - 1]
        u_aux_rs = 1*u_rs[i][t - 1]
        
        a_rs = rs_params['a']
        b_rs = rs_params['b']
        c_rs = rs_params['c'] + 15*random_factor**2
        d_rs = rs_params['d'] - 6*random_factor**2
                
        if (v_aux_rs >= vp):
            AP_aux_rs = 1
            v_aux_rs = v_rs[i][t]
            v_rs[i][t] = c_rs
            u_rs[i][t] = u_aux_rs + d_rs
        else:
            AP_aux_rs = 0
            dv_rs = izhikevich_dvdt(v = v_aux_rs, u = u_aux_rs, I = I_rs)
            du_rs = izhikevich_dudt(v = v_aux_rs, u = u_aux_rs, a = a_rs, b = b_rs)
        
            v_rs[i][t] = v_aux_rs + dt*dv_rs
            u_rs[i][t] = u_aux_rs + dt*du_rs
            
        for j in range(p):
            # variable just after the spike
            next_u_rs = u_E[0][j] + U_E[j]*(1 - u_E[0][j]) 
            # u -> utilization factor -> resources ready for use
            u_E[0][j] = u_E[0][j] + dt*(-u_E[0][j]/t_f_E[j] + U_E[j]*(1 - u_E[0][j])*AP_aux_rs)
            # x -> availabe resources -> Fraction of resources that remain available after neurotransmitter depletion
            R_E[0][j] = R_E[0][j] + dt*((1 - R_E[0][j])/t_d_E[j] - next_u_rs*R_E[0][j]*AP_aux_rs)
            # PSC
            I_E[0][j] = I_E[0][j] + dt*(-I_E[0][j]/t_s_E + A_E[j]*R_E[0][j]*next_u_rs*AP_aux_rs)
        
        PSC_rs[i][t] = np.sum(I_E)

    # FS NEURON - Inhibitory
    AP_aux_fs = 0
    for i in range(n_rs):
        v_aux_fs = 1*v_fs[i][t - 1]
        u_aux_fs = 1*u_fs[i][t - 1]
        
        a_fs = fs_params['a'] + 0.08*random_factor
        b_fs = fs_params['b'] - 0.05*random_factor
        c_fs = fs_params['c']
        d_fs = fs_params['d']
        
        if (v_aux_fs >= vp):
            AP_aux_fs = 1
            v_aux_fs = v_fs[i][t]
            v_fs[i][t] = c_fs
            u_fs[i][t] = u_aux_fs + d_fs
        else:
            AP_aux_fs = 0
            dv = izhikevich_dvdt(v = v_aux_fs, u = u_aux_fs, I = I_fs)
            du = izhikevich_dudt(v = v_aux_fs, u = u_aux_fs, a = a_fs, b = b_fs)
            
            rs_contribution = PSC_rs[i][t]*-5e2/2.5
            # rs_contribution = 0
        
            v_fs[i][t] = v_aux_fs + dt*(dv + rs_contribution)
            u_fs[i][t] = u_aux_fs + dt*du
            
        fs_syn[i][t] = tm_synapse_eq(u=u_E, R=R_E, I=I_E, AP=AP_aux_fs, t_f=t_f_E, t_d=t_d_E, t_s=t_s_E, U=U_E, A=A_E, dt=dt)['Ipost']
            

print_signal(v_rs[0], "Izhikevich model - Regular Spiking")
print_signal(v_fs[0], "Izhikevich model - Fast Spiking")

print_signal(PSC_rs, "synapse - Regular Spiking")
print_signal(fs_syn, "synapse - Fast Spiking")

# =============================================================================
# MAIN - STF
# =============================================================================
for t in time:
    # RS NEURON - Excitatory
    AP_aux_rs = 0
    for i in range(n_rs):
        v_aux_rs = 1*v_rs[i][t - 1]
        u_aux_rs = 1*u_rs[i][t - 1]
        
        a_rs = rs_params['a']
        b_rs = rs_params['b']
        c_rs = rs_params['c'] + 15*random_factor**2
        d_rs = rs_params['d'] - 6*random_factor**2
                
        if (v_aux_rs >= vp):
            AP_aux_rs = 1
            v_aux_rs = v_rs[i][t]
            v_rs[i][t] = c_rs
            u_rs[i][t] = u_aux_rs + d_rs
        else:
            AP_aux_rs = 0
            dv_rs = izhikevich_dvdt(v = v_aux_rs, u = u_aux_rs, I = I_rs)
            du_rs = izhikevich_dudt(v = v_aux_rs, u = u_aux_rs, a = a_rs, b = b_rs)
        
            v_rs[i][t] = v_aux_rs + dt*dv_rs
            u_rs[i][t] = u_aux_rs + dt*du_rs
            
        for j in range(p):
            # variable just after the spike
            next_u_rs = u_E[0][j] + U_E[j]*(1 - u_E[0][j]) 
            # u -> utilization factor -> resources ready for use
            u_E[0][j] = u_E[0][j] + dt*(-u_E[0][j]/t_f_E[j] + U_E[j]*(1 - u_E[0][j])*AP_aux_rs)
            # x -> availabe resources -> Fraction of resources that remain available after neurotransmitter depletion
            R_E[0][j] = R_E[0][j] + dt*((1 - R_E[0][j])/t_d_E[j] - next_u_rs*R_E[0][j]*AP_aux_rs)
            # PSC
            I_E[0][j] = I_E[0][j] + dt*(-I_E[0][j]/t_s_E + A_E[j]*R_E[0][j]*next_u_rs*AP_aux_rs)
        
        PSC_rs[i][t] = np.sum(I_E)
            
    # LTS NEURON - Inhibitory
    AP_aux_lts = 0
    for i in range(n_lts):
        v_aux_lts = 1*v_lts[i][t - 1]
        u_aux_lts = 1*u_lts[i][t - 1]
        
        a_lts = lts_params['a'] + 0.08*random_factor
        b_lts = lts_params['b'] - 0.05*random_factor
        c_lts = lts_params['c']
        d_lts = lts_params['d']
        
        if (v_aux_lts >= vp):
            AP_aux_lts = 1
            v_aux_lts = v_lts[i][t]
            v_lts[i][t] = c_lts
            u_lts[i][t] = u_aux_lts + d_lts
        else:
            AP_aux_lts = 0
            dv = izhikevich_dvdt(v = v_aux_lts, u = u_aux_lts, I = I_lts)
            du = izhikevich_dudt(v = v_aux_lts, u = u_aux_lts, a = a_lts, b = b_lts)
        
            rs_contribution = PSC_rs[i][t]*-5e2/2.5
            
            v_lts[i][t] = v_aux_lts + dt*(dv + rs_contribution)
            u_lts[i][t] = u_aux_lts + dt*du
    
        lts_syn[i][t] = tm_synapse_eq(u=u_E, R=R_E, I=I_E, AP=AP_aux_lts, t_f=t_f_E, t_d=t_d_E, t_s=t_s_E, U=U_E, A=A_E, dt=dt)['Ipost']

print_signal(v_rs[0], "Izhikevich model - Regular Spiking")
print_signal(v_lts[0], "Izhikevich model - Low Threshold Spiking")

print_signal(PSC_rs, "synapse - Regular Spiking")
print_signal(lts_syn, "synapse - Low Threshold Spiking")