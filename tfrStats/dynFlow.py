# subject index
i_sub = 0

# hemis index
i_hemis = 0

# cond index
i_cond = 0 # 10 conditions

# ROIs index    
n_rois = len(np.unique(ROIs))
EC     = np.zeros((n_rois,n_rois))
tau    = np.zeros((n_rois))

    
# Dyn communicability = flow with unit perturbation at each node
# Define the temporal resolution for the "simulation"

tfinal = 30
dt = 0.5
tpoints = np.arange(0.0, tfinal+dt, dt)
nsteps = len(tpoints)
print(nsteps)
dynflow_EC = np.zeros((J_mod.shape[0],3,2,nsteps,n_rois,n_rois))

for i_sub in range(J_mod.shape[0]):
    for i_cond in range(3):
        runs_idx = conf['runs'][i_cond]
        for i_hemis in range(2):
            
            hemis = conf['hemis'][i_hemis]
            #print(hemis)
            if hemis == 'left':
                hemis_idx = np.arange(0,12)
            if hemis ==   'right':
                hemis_idx = np.arange(12,24)

            ROIs_ = ROIs[hemis_idx]
            tau_x_ = tau_x[:,:,hemis_idx]

            for i_roi in range(n_rois):

                # Get Jacobian
                J       = np.nanmean(J_mod[i_sub,runs_idx,:,:], axis=0)
                #print(J.shape)

                # Get a time-constant from Jacobian
                #print(tau_x_.shape)
                k =  np.asarray(runs_idx)
                #print(k.shape)
                t = np.nanmean(tau_x_[i_sub,k, :],axis=0)
                t = t[ROIs_== i_roi]
                tau[i_roi] = np.nanmean(t.flatten()) 

                # Set the matrix of noisy inputs
                S = np.nanmean(Sigma_mod[i_sub,runs_idx,:,:], axis=0)


                for j_roi in range(n_rois):


                    if i_roi == j_roi:
                        # Lump the diagonal elements
                        J_diag   = J[np.ix_(hemis_idx,hemis_idx)]
                        flow = J_diag[ROIs_== i_roi, ROIs_==j_roi]
                        EC[i_roi, j_roi] =  np.nanmean(flow.flatten())
                    else:
                        # Lump off-diagonal
                        J_       = J[np.ix_(hemis_idx,hemis_idx)]
                        np.fill_diagonal(J_, 'nan')
                        flow = J_[ROIs_== i_roi, ROIs_==j_roi]
                        EC[i_roi, j_roi] =  np.nanmean(flow.flatten())    

            # Diagonal of the Jacobian
            J0 = -np.eye(n_rois) / tau
            J = J0 + EC

            S = np.eye(n_rois)

            # Calculate the dynamic flow for a time span between 0 and tmax
            hola = ndf.DynFlow(EC, tau, S, tmax=tfinal, timestep=dt, normed=True)
            #print(hola.shape)
            dynflow_EC[i_sub,i_cond,i_hemis,:,:,:] = hola

print('listo')