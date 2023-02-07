"""
@author: Celine Soeiro

@description: Tsodyks and Markram Synapse Model according FL

For an individual synapse the model reproduces the postsynaptic responses enerated 
by any presynaptice spike train t_sp for interpyramidal synapses in layer V
    
* PHENOMENOLOGICAL MODEL:
    dudt = -(u/t_f) + U*(1 - u)*delta(t - t_sp - delta)
    dxdt = -((1 - x)/t_d) - u*x*delta(t - t_sp - delta)    
    dIdt = -(I/t_sp) + A*u*x*delta(t - t_sp - delta)
    
Where:    
    t_sp -> time in which the presynaptic spike arrives
    delta -> Dirac delta function
    U -> increment of u produced by an incoming spike
    t_f -> decay time constant of variable u
    t_d -> recovery time constant of variable x
    t_s -> decay time constant of variable I
    A -> synaptic response amplitude that would be produced with the release of all neurotransmitters (absolute synaptic response)

The two major parameters of the model are A and U
    A -> Can be exhibited only by activating all of the resources
    U -> Determines the dynamics of the synaptic response
    

"""

import numpy as np

def getParamaters(synapse_type: str):
    if (synapse_type == 'excitatory'):
        return {
            # [Facilitating, Depressing, Pseudo-linear]
            't_f': [670, 17, 326],
            't_d': [138, 671, 329],
            'U': [0.09, 0.5, 0.29],
            'distribution': [0.2, 0.63, 0.17],
        };
    elif (synapse_type == 'inhibitory'):
        return {
            # [Facilitating, Depressing, Pseudo-linear]
            't_f': [376, 21, 62],
            't_d': [45, 706, 144],
            'U': [0.016, 0.25, 0.32],
            'distribution': [0.08, 0.75, 0.17],
        };
    
    else:
        return 'Invalid synapse_type. Synapse_type must be excitatory or inhibitory.'
    
# =============================================================================
# Equations
# =============================================================================
    
def x_eq(x, t_d, u, delta):
    # fraction of the neurotransmitter resources that remain available after synaptic transmission
    return -(1-x)/t_d - u*x*delta

def u_eq(u, t_f, U, delta):
    # fraction of available neurotransmitter resources ready to be used
    return -(u/t_f) + U*(1 - u)*delta
    
def I_eq(I, t_s, A, u, x, delta):
    # post-synaptic current
    return -(I/t_s) + A*u*x*delta

# =============================================================================
# Synapse
# =============================================================================
    
def TM_Synapse(
        t_event: float, 
        n_sim: int, 
        t_delay: int, 
        dt: float, 
        synapse_type: str, 
        dbs = False,
    ):
    t = np.arange(t_delay + 1, n_sim - 1)
    n_sim = int(n_sim)
    
    # parameters
    tau_f = getParamaters(synapse_type)['t_f']
    tau_d = getParamaters(synapse_type)['t_d']
    U = getParamaters(synapse_type)['U']
    A = getParamaters(synapse_type)['distribution']
    
    parameters_length = len(tau_f)
    
    if (synapse_type == 'inhibitory'):
        tau_s = 11
    elif (synapse_type == 'excitatory'):
        tau_s = 3

    # Initial values
    u = np.zeros((3, n_sim))
    x = np.ones((3, n_sim))
    I = np.zeros((3, n_sim))
    
    # poissonian contribution to the spike
    spd = np.zeros((1,n_sim))

    # Loop trhough the parameters
    for p in range(parameters_length):
        # Update the variable
        if (dbs == False):
            spd[0][t_event] = 1/dt
        # Solve ODE using Euler method
        for i in range(t_delay + 1, n_sim - 1):
            if (dbs):
                delta = t_event[0][i - t_delay]
            else:
                delta = spd[0][i - t_delay] # marks when the spike occurs
            
            u[p][i + 1] = u[p][i] + dt*u_eq(u[p][i], tau_f[p], U[p], delta)
            x[p][i + 1] = x[p][i] + dt*x_eq(x[p][i], tau_d[p], u[p][i], delta)
            I[p][i + 1] = I[p][i] + dt*I_eq(I[p][i], tau_s, A[p], u[p][i], x[p][i], delta)

    if (dbs):
        I_post_synaptic = I.sum(axis=0)
    else:
        I_post_synaptic = np.concatenate(I, axis=None)
    
    return t, I_post_synaptic

# =============================================================================
# Instantaneos Synapse
# =============================================================================

def TM_Synapse_Inst(
        dt: float, 
        synapse_type: str, 
        delta: float, 
        n_sim: int,
        layer = None,
    ):
    # Turn number of simulations into an int
    n_sim = int(n_sim)
    
    # get parameters
    tau_f = getParamaters(synapse_type)['t_f']
    tau_d = getParamaters(synapse_type)['t_d']
    U = getParamaters(synapse_type)['U']
    A = getParamaters(synapse_type)['distribution']
    parameters_length = len(tau_f)
    
    # Pick tau_s according to synapse time
    if (synapse_type == 'inhibitory'):
        tau_s = 11
    elif (synapse_type == 'excitatory'):
        tau_s = 3
        
    # If in any specific layer, set the distribution according to the layer
    if (layer == 'D'):
        A = [0,1,0]*np.ones((1,3))
    elif (layer == 'F'):
        A = [1,0,0]*np.ones((1,3))

    # Initial values
    u = np.zeros((3, n_sim))
    x = np.ones((3, n_sim))
    I = np.zeros((3, n_sim))
    
    # Loop trhough the parameters
    for p in range(parameters_length - 1):
        # Solve EDOs using Euler method 
        u[p + 1] = u[p] + dt*u_eq(u[p], tau_f[p], U[p], delta)
        x[p + 1] = x[p] + dt*x_eq(x[p], tau_d[p], u[p], delta)
        I[p + 1] = I[p] + dt*I_eq(I[p], tau_s, A[p], u[p], x[p], delta)
        
    # Concatenate the final current
    I_post_synaptic = np.concatenate(I, axis=None)
    
    return u, x, I, I_post_synaptic






