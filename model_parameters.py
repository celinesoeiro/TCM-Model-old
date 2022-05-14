"""
Created on Wed May 11 22:03:39 2022

@author: Celine Soeiro

PD -> Parkinson Desease

"""

import numpy as np

def model_parameters():
    simulation_time = 10
    number_trials = 1
    dt = 0.1
    fs = 1000/dt
    dbs_on = 5*67
    dbs_off = 0
    synaptic_fidelity = dbs_off
    sim_time = 10
    sim_time_ms = (sim_time + 1)*1000
    sim_steps = np.round(sim_time_ms/dt)
    td_synapse = 1
    td_thalamus_cortex = 15 # 25
    td_cortex_thalamus = 20
    hyperdirect_neurons = 0.1
    connectivity_factor_neurons = 2.5
    connectivity_factor_PD = 5
    
    if (td_thalamus_cortex >= td_cortex_thalamus): 
        t_vec = np.arange(td_thalamus_cortex + td_synapse + 1, sim_steps)
    else:
        t_vec = np.arange(td_cortex_thalamus + td_synapse + 1, sim_steps)
    
    neuron_quantities = {
        'qnt_neurons_s': 100,
        'qnt_neurons_m': 100,
        'qnt_neurons_d': 100,
        'qnt_neurons_ci': 100,
        'qnt_neurons_tc': 100,
        'qnt_neurons_tr': 40,
        'qnt_neurons_hyperdirect': 100*hyperdirect_neurons, # Layer D neurons (qnt_neurons_d)
        'total': 100 + 100 + 100 + 100 + 100 + 40,
        }
    
    neuron_per_structure = {
        'neurons_s_regular_spiking': 0.5*neuron_quantities['qnt_neurons_s'],
        'neurons_s_intrinsically_bursting': 0.5*neuron_quantities['qnt_neurons_s'],
        'neurons_m_regular_spiking': 1*neuron_quantities['qnt_neurons_m'],
        'neurons_d_regular_spiking': 0.7*neuron_quantities['qnt_neurons_d'],
        'neurons_d_intrinsically_bursting': 0.3*neuron_quantities['qnt_neurons_d'],
        'neurons_c_fast_spiking': 0.5*neuron_quantities['qnt_neurons_ci'],
        'neurons_d_low_threshold_spiking': 0.5*neuron_quantities['qnt_neurons_ci'],
        'neurons_tcr_tc': 1*neuron_quantities['qnt_neurons_tcr'],
        'neurons_tcr_tr': 1*neuron_quantities['qnt_neurons_trn'],
        }
    
    TC_model = {
        'number_trials': number_trials,
        'synaptic_fidelity': synaptic_fidelity, # DBS off - To turn DBS on set this value to 5*67
        'hyperdirect_neurons': hyperdirect_neurons, # Percentage of PNs affected in D by DBS
        'simulation_time': simulation_time, # simulation time in seconds
        'dt': dt, # time step
        'sampling_frequency': fs, # in Hz
        'simulation_steps': sim_steps,
        'chop_till': 1*fs, # cut the first 1s of simulation
        'time_delay_between_layers': 8,
        'time_delay_within_layers': 1,
        'time_delay_thalamus_cortex': 15,
        'time_delay_cortex_thalamus': 20,
        'transmission_delay_synapse': 1,
        'time_vector': t_vec,
        'connectivity_factor_neurons': connectivity_factor_neurons,
        'connectivity_factor_PD': connectivity_factor_PD,
        }
    
    synaptic_fidelity = {
        'synaps_fidelity_CI': 1*synaptic_fidelity,
        'synapse_fidelity_M': 0*synaptic_fidelity,
        'synapse_fidelity_S': 1*synaptic_fidelity,
        'synapse_fidelity_R': 1*synaptic_fidelity,
        'synapse_fidelity_N': 1*synaptic_fidelity,
        }
    
    
    coupling_matrix_normal = {
        ''
        }
    
    
    return neuron_quantities, neuron_per_structure, TC_model, synaptic_fidelity
    

def coupling_matrix_normal(facilitating_factor, n_s, n_m, n_d, n_ci, n_tc, n_tr):
    division_factor = facilitating_factor
    
    initial = 0
    final = 1
    interval = final - initial
    
    # =============================================================================
    #     These are to restrict the normalized distribution variance or deviation from the mean
    # =============================================================================
    r_s = initial + interval*np.random.rand(n_s, 1)
    r_m = initial + interval*np.random.rand(n_m, 1)
    r_d = initial + interval*np.random.rand(n_d, 1)
    r_ins = initial + interval*np.random.rand(n_ci, 1)
    r_tc = initial + interval*np.random.rand(n_tc, 1)
    r_tr = initial + interval*np.random.rand(n_tr, 1)
    
    # =============================================================================
    #     COUPLING STRENGTHs within each structure (The same in Normal and PD)
    # EE -> Excitatory to Excitatory (?) 
    # II -> Inhibitory to Inhibitory (?)
    # =============================================================================
    ## Layer S (was -1e-2 for IEEE paper)
    aee_s = -1e1/division_factor
    W_EE_s = aee_s*r_s
    ## Layer M (was -1e-2 for IEEE paper)
    aee_m = -1e1/division_factor
    W_EE_m = aee_m*r_m
    ## Layer D (was -1e-2 for IEEE paper)
    aee_d = -1e1/division_factor
    W_EE_d = aee_d*r_d
    ## INs 
    aii_INs = -5e2/division_factor
    W_II_ins = aii_INs*r_ins
    ## Reticular cells
    aii_ret = -5e1/division_factor
    W_II_ret = aii_ret*r_tc
    ## Relay cells
    aee_rel = 0/division_factor
    W_EE_rel = aee_rel*r_tr
    
    # =============================================================================
    #     COUPLING STRENGTHs between structures
    # =============================================================================
    #     S
    # =============================================================================
    # M to S coupling
    aee_sm = 1e1/division_factor
    W_EE_s_m = aee_sm*r_s
    # D to S coupling
    aee_ds = 5e2/division_factor
    W_EE_s_d = aee_ds*r_s
    # CI (INs) to S coupling
    aei_sINs = -5e2/division_factor
    W_EI_s_INs = aei_sINs*r_s
    # Reticular to S coupling
    aei_sRet = 0/division_factor      
    W_EI_s_Ret = aei_sRet*r_s
    # Rel. to S couplings
    aee_sRel = 0/division_factor      
    W_EE_s_Rel = aee_sRel*r_s         
    # =============================================================================
    #     M
    # =============================================================================
    # S to M
    aee_ms = 3e2/facilitating_factor      
    W_EE_m_s = aee_ms*r_m             
    # D to M couplings
    aee_md = 0/facilitating_factor        
    W_EE_m_d = aee_md*r_m             
    # INs to M couplings
    aei_mINs = -3e2/facilitating_factor    
    W_EI_m_INs = aei_mINs*r_m         
    # Ret. to M couplings    
    aei_mRet = 0/facilitating_factor      
    W_EI_m_Ret = aei_mRet*r_m      
    # Rel. to M couplings
    aee_mRel = 0/facilitating_factor      
    W_EE_m_Rel = aee_mRel*r_m
    # =============================================================================
    #     D
    # =============================================================================
    # S to D couplings
    aee_ds = 3e2/facilitating_factor      
    W_EE_d_s = aee_ds*r_d
    # M to D couplings
    aee_dm = 0/facilitating_factor        
    W_EE_d_m = aee_dm*r_d
    # INs to D couplings
    aei_dINs = -7.5e3/facilitating_factor   
    W_EI_d_INs = aei_dINs*r_d
    # Ret. to D couplings
    aei_dRet = 0/facilitating_factor      
    W_EI_d_Ret = aei_dRet*r_d
    # Rel. to D couplings
    aee_dRel = 1e1/facilitating_factor    
    W_EE_d_Rel = aee_dRel*r_d
    # =============================================================================
    #     INs (CI)
    # =============================================================================
    # S to INs couplings
    aie_inss = 2e2/facilitating_factor    
    W_IE_INs_s = aie_inss*r_ins
    # M to INs couplings
    aie_insm = 2e2/facilitating_factor    
    W_IE_INs_m = aie_insm*r_ins
    # D to INs couplings
    aie_insd = 2e2/facilitating_factor    
    W_IE_INs_d = aie_insd*r_ins
    # Ret. to INs couplings
    aii_InsRet = 0/facilitating_factor    
    W_II_INs_Ret = aii_InsRet*r_ins
    # Rel. to INs couplings
    aie_InsRel = 1e1/facilitating_factor  
    W_IE_INs_Rel = aie_InsRel*r_ins
    # =============================================================================
    #     Reticular
    # =============================================================================
    # S to Ret couplings
    aie_rets = 0/facilitating_factor      
    W_IE_Ret_s = aie_rets*r_tc     
    # M to Ret couplings
    aie_retm = 0/facilitating_factor      
    W_IE_Ret_m = aie_retm*r_tc     
    # D to Ret couplings
    aie_retd = 7e2/facilitating_factor   
    W_IE_Ret_d = aie_retd*r_tc;     
    # Ret. Ret INs couplings
    aii_RetIns = 0/facilitating_factor    
    W_II_Ret_INs = aii_RetIns*r_tc 
    # Rel. Ret INs couplings
    aie_RetRel = 1e3/facilitating_factor  
    W_IE_Ret_Rel = aie_RetRel*r_tc 
    # =============================================================================
    #     Rele
    # =============================================================================
    # S to Rel couplings
    aee_rels = 0/facilitating_factor     
    W_EE_Rel_s = aee_rels*r_tr       
    # M to Rel couplings
    aee_relm = 0/facilitating_factor      
    W_EE_Rel_m = aee_relm*r_tr 	    
    # D to Rel couplings
    aee_reld = 7e2/facilitating_factor
    W_EE_Rel_d = aee_reld*r_tr       
    # INs to Rel couplings
    aei_RelINs = 0/facilitating_factor    
    W_EI_Rel_INs = aei_RelINs*r_tr   
    # Ret to Rel couplings
    aei_RelRet = -5e2/facilitating_factor 
    W_EI_Rel_Ret = aei_RelRet*r_tr   
    
    weights = {
        'W_EE_s': W_EE_s,
        'W_EE_m': W_EE_m,
        'W_EE_d': W_EE_d,
        'W_II_ins': W_II_ins,
        'W_II_ret': W_II_ret,
        'W_EE_rel': W_EE_rel,
        'W_EE_s_m': W_EE_s_m,
        'W_EE_s_d': W_EE_s_d,
        'W_EI_s_INs': W_EI_s_INs,
        'W_EI_s_Ret': W_EI_s_Ret,
        'W_EE_s_Rel': W_EE_s_Rel,
        'W_EE_m_s': W_EE_m_s,
        'W_EE_m_d': W_EE_m_d,
        'W_EI_m_INs': W_EI_m_INs,
        'W_EI_m_Ret': W_EI_m_Ret,
        'W_EE_m_Rel': W_EE_m_Rel,
        'W_EE_d_s': W_EE_d_s,
        'W_EE_d_m': W_EE_d_m,
        'W_EI_d_INs': W_EI_d_INs,
        'W_EI_d_Ret': W_EI_d_Ret,
        'W_EE_d_Rel': W_EE_d_Rel,
        'W_IE_INs_s': W_IE_INs_s,
        'W_IE_INs_m': W_IE_INs_m,
        'W_IE_INs_d': W_IE_INs_d,
        'W_II_INs_Ret': W_II_INs_Ret,
        'W_IE_INs_Rel': W_IE_INs_Rel,
        'W_IE_Ret_s': W_IE_Ret_s,
        'W_IE_Ret_m': W_IE_Ret_m,
        'W_IE_Ret_d': W_IE_Ret_d,
        'W_II_Ret_INs': W_II_Ret_INs,
        'W_IE_Ret_Rel': W_IE_Ret_Rel,
        'W_EE_Rel_s': W_EE_Rel_s,
        'W_EE_Rel_m': W_EE_Rel_m,
        'W_EE_Rel_d': W_EE_Rel_d,
        'W_EI_Rel_IN_s': W_EI_Rel_INs,
        'W_EI_Rel_Ret': W_EI_Rel_Ret,
        }
    
    return weights
    
    
def coupling_matrix_PD(facilitating_factor, n_s, n_m, n_d, n_ci, n_tc, n_tr):
    division_factor = facilitating_factor
    
    initial = 0
    final = 1
    interval = final - initial
    
    # =============================================================================
    #     These are to restrict the normalized distribution variance or deviation from the mean
    # =============================================================================
    r_s = initial + interval*np.random.rand(n_s, 1)
    r_m = initial + interval*np.random.rand(n_m, 1)
    r_d = initial + interval*np.random.rand(n_d, 1)
    r_ins = initial + interval*np.random.rand(n_ci, 1)
    r_tc = initial + interval*np.random.rand(n_tc, 1)
    r_tr = initial + interval*np.random.rand(n_tr, 1)
    
    # =============================================================================
    #     COUPLING STRENGTHs within each structure (The same in Normal and PD)
    # EE -> Excitatory to Excitatory (?) 
    # II -> Inhibitory to Inhibitory (?)
    # =============================================================================
    ## Layer S (was -1e-2 for IEEE paper)
    aee_s = -1e1/division_factor
    W_EE_s = aee_s*r_s
    ## Layer M (was -1e-2 for IEEE paper)
    aee_m = -1e1/division_factor
    W_EE_m = aee_m*r_m
    ## Layer D (was -1e-2 for IEEE paper)
    aee_d = -1e1/division_factor
    W_EE_d = aee_d*r_d
    ## INs 
    aii_INs = -5e2/division_factor
    W_II_ins = aii_INs*r_ins
    ## Reticular cells
    aii_ret = -5e1/division_factor
    W_II_ret = aii_ret*r_tc
    ## Relay cells
    aee_rel = 0/division_factor
    W_EE_rel = aee_rel*r_tr
    
    # =============================================================================
    #     COUPLING STRENGTHs between structures
    # =============================================================================
    #     S
    # =============================================================================
    # M to S couplings
    aee_sm = 3e2/division_factor  
    W_EE_s_m = aee_sm*r_s             
    # D to S couplings
    aee_sd = 5e2/division_factor      
    W_EE_s_d = aee_sd*r_s          
    # INs to S couplings
    aei_sINs = -7.5e2/division_factor    
    W_EI_s_INs = aei_sINs*r_s         
    # Ret. to S couplings
    aei_sRet = 0/division_factor      
    W_EI_s_Ret = aei_sRet*r_s         
    # Rel. to S couplings
    aee_sRel = 0/division_factor    
    W_EE_s_Rel = aee_sRel*r_s         
    # =============================================================================
    #     M
    # =============================================================================
    # S to M couplings
    aee_ms = 1e1/division_factor    
    W_EE_m_s = aee_ms*r_m            
    # D to M couplings
    aee_md = 0/division_factor        
    W_EE_m_d = aee_md*r_m            
    # INs to M couplings
    aei_mINs = -7.5e2/division_factor    
    W_EI_m_INs = aei_mINs*r_m      
    # Ret. to M couplings
    aei_mRet = 0/division_factor      
    W_EI_m_Ret = aei_mRet*r_m         
    # Rel. to M couplings
    aee_mRel = 0/division_factor      
    W_EE_m_Rel = aee_mRel*r_m         
    # =============================================================================
    #     D
    # =============================================================================
    # M to S couplings
    aee_ds = 3e2/facilitating_factor    
    W_EE_d_s = aee_ds*r_d             
    # M to D couplings
    aee_dm = 0/facilitating_factor    
    W_EE_d_m = aee_dm*r_d          
    # INs to D couplings
    aei_dINs = -5e3/facilitating_factor   
    W_EI_d_INs = aei_dINs*r_d         
    # Ret. to D couplings
    aei_dRet = 0/facilitating_factor
    W_EI_d_Ret = aei_dRet*r_d
    # Rel. to D couplings
    aee_dRel = 1e3/facilitating_factor
    W_EE_d_Rel = aee_dRel*r_d         
    # =============================================================================
    #     INs
    # =============================================================================
    # S to INs couplings
    aie_inss = 2e2/facilitating_factor    
    W_IE_INs_s = aie_inss*r_ins
    # M to INs couplings
    aie_insm = 2e2/facilitating_factor    
    W_IE_INs_m = aie_insm*r_ins     
    # D to INs couplings
    aie_insd = 2e2/facilitating_factor    
    W_IE_INs_d = aie_insd*r_ins  
    # Ret. to INs couplings
    aii_InsRet = 0/facilitating_factor   
    W_II_INs_Ret = aii_InsRet*r_ins 
    # Rel. to INs couplings
    aie_InsRel = 1e3/facilitating_factor  
    W_IE_INs_Rel = aie_InsRel*r_ins
    # =============================================================================
    #     Reticular
    # =============================================================================
    # S to Ret couplings
    aie_rets = 0/facilitating_factor      
    W_IE_Ret_s = aie_rets*r_tc     
    # M to Ret couplings
    aie_retm = 0/facilitating_factor     
    W_IE_Ret_m = aie_retm*r_tc     
    # D to Ret couplings
    aie_retd = 1e2/facilitating_factor    
    W_IE_Ret_d = aie_retd*r_tc     
    # Ret. Ret INs couplings
    aii_RetIns = 0/facilitating_factor
    W_II_Ret_INs = aii_RetIns*r_tc 
    # Rel. Ret INs couplings
    aie_RetRel = 5e2/facilitating_factor  
    W_IE_Ret_Rel = aie_RetRel*r_tc 
    # =============================================================================
    #     Rele
    # =============================================================================
    # S to Rel couplings
    aee_rels = 0/facilitating_factor      
    W_EE_Rel_s = aee_rels*r_tr       
    # M to Rel couplings
    aee_relm = 0/facilitating_factor      
    W_EE_Rel_m = aee_relm*r_tr 	    
    # D to Rel couplings
    aee_reld = 1e2/facilitating_factor    
    W_EE_Rel_d = aee_reld*r_tr       
    # INs to Rel couplings
    aei_RelINs = 0/facilitating_factor
    W_EI_Rel_INs = aei_RelINs*r_tr   
    # Ret to Rel couplings
    aei_RelRet = -2.5*1e3/facilitating_factor
    W_EI_Rel_Ret = aei_RelRet*r_tr   
    
    weights = {
        'W_EE_s': W_EE_s,
        'W_EE_m': W_EE_m,
        'W_EE_d': W_EE_d,
        'W_II_ins': W_II_ins,
        'W_II_ret': W_II_ret,
        'W_EE_rel': W_EE_rel,
        'W_EE_s_m': W_EE_s_m,
        'W_EE_s_d': W_EE_s_d,
        'W_EI_s_INs': W_EI_s_INs,
        'W_EI_s_Ret': W_EI_s_Ret,
        'W_EE_s_Rel': W_EE_s_Rel,
        'W_EE_m_s': W_EE_m_s,
        'W_EE_m_d': W_EE_m_d,
        'W_EI_m_INs': W_EI_m_INs,
        'W_EI_m_Ret': W_EI_m_Ret,
        'W_EE_m_Rel': W_EE_m_Rel,
        'W_EE_d_s': W_EE_d_s,
        'W_EE_d_m': W_EE_d_m,
        'W_EI_d_INs': W_EI_d_INs,
        'W_EI_d_Ret': W_EI_d_Ret,
        'W_EE_d_Rel': W_EE_d_Rel,
        'W_IE_INs_s': W_IE_INs_s,
        'W_IE_INs_m': W_IE_INs_m,
        'W_IE_INs_d': W_IE_INs_d,
        'W_II_INs_Ret': W_II_INs_Ret,
        'W_IE_INs_Rel': W_IE_INs_Rel,
        'W_IE_Ret_s': W_IE_Ret_s,
        'W_IE_Ret_m': W_IE_Ret_m,
        'W_IE_Ret_d': W_IE_Ret_d,
        'W_II_Ret_INs': W_II_Ret_INs,
        'W_IE_Ret_Rel': W_IE_Ret_Rel,
        'W_EE_Rel_s': W_EE_Rel_s,
        'W_EE_Rel_m': W_EE_Rel_m,
        'W_EE_Rel_d': W_EE_Rel_d,
        'W_EI_Rel_IN_s': W_EI_Rel_INs,
        'W_EI_Rel_Ret': W_EI_Rel_Ret,
        }
    
    return weights