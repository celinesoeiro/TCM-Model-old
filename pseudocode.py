for t in time:
    I_synapse = zeros(1,neurons)
    
    for n in neurons:
        v_aux = v[n][t - 1]
        u_aux = u[n][t - 1]
        AP_aux = 0
        
        if(v_aux >= vp + a_noise[t - 1]):
            AP_aux = 1
            AP[n][t] = t - 1
            v_aux = v[n][t]
            v[n][t] = c[0][n]
            u[n][t] = u_aux + d[0][n]
        else:
            AP[n][t] = 0
            AP_aux = 0
            
            # Coupling S 
            coupling_S = W_S[n][0]*1*PSC_S[0][t - td_bl - td_syn - 1]
            # Coupling M  
            coupling_M = W_M[n][0]*1*PSC_M[0][t - td_bl - td_syn - 1]
            # Coupling D
            coupling_D = W_D[n][0]*1*PSC_D[0][t - td_wl - td_syn - 1]
            # Coupling CI  
            coupling_CI = W_CI[n][0]*1*PSC_CI[0][t - td_wl - td_syn - 1]
            # Coupling TC 
            coupling_TC = W_TC[n][0]*1*PSC_TC[0][t - td_tc - td_syn - 1]
            # Coupling TR 
            coupling_TR = W_TR[n][0]*1*PSC_TR[0][t - td_tc - td_syn - 1]
            
            dv = izhikevich_dvdt(v = v_aux, u = u_aux, I = I[n])
            du = izhikevich_dudt(v = v_aux, u = u_aux, a = a[0][n], b = b[0][n])
            
            coupling_cortex = (coupling_S + coupling_M + coupling_D + coupling_CI)/neurons
            coupling_thalamus = (coupling_TC + coupling_TR)/neurons
            bg_activity = b_noise[n][t - 1] + I_ps[0][t - td_wl - td_syn - 1] - I_ps[1][t - td_wl - td_syn - 1]
        
            v[n][t] = v_aux + dt*(dv + coupling_cortex + coupling_thalamus + bg_activity)
            u[n][t] = u_aux + dt*du
            
        # Synapse - Within cortex  
        syn = tm_synapse_eq(u = u_syn, R = R_syn, I = I_syn, AP = AP_aux, t_f = t_f, t_d = t_d, t_s = t_s, U = U, A = A, dt = dt, p = p)
        
        R_syn = 1*syn['R']
        u_syn = 1*syn['u']
        I_syn = 1*syn['I']
        I_synapse[0][n] = 1*syn['Ipost']
    
    PSC[0][t] = sum(I_synapse)



        