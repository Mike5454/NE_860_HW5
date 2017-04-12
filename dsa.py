# To correct the division in the term D we must import division
from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import scipy.sparse.linalg

def build_diffusion_matrix(Sigma_t, Sigma_s, N, dx):
    Sigma_r = Sigma_t - Sigma_s
    # find diffusion coefficient
    D = 1/(3*Sigma_t)
    # Use sparse matrix operator
    L = csr_matrix((N, N))
    # Left boundary
    L[0, 0] = Sigma_r + 2*D/((4*D+dx)*dx)
   # L[0, 1] = -2*D/((4*D+dx)*dx)
    
    # Right boundary
    L[-1, -1] = Sigma_r + 2*D/((4*D+dx)*dx)

    # Central
    D_tilde = 2*D*D/((dx*D+dx*D)*dx)
    for i in range(1, N-1) :
        L[i, i+1] = -D_tilde/dx
        L[i, i-1] = -D_tilde/dx
        L[i, i]   = 2*D_tilde/dx + Sigma_r
    return L
         
# Solve using DSA
def wg_dsa(v, Sigma_t, Sigma_s, N, dx) :
    L = build_diffusion_matrix(Sigma_t, Sigma_s, N, dx)
    return v + Sigma_s*np.linalg.solve(L, v)

# Call definitions to initiate solution
if __name__ == "__main__" :
    cvals = np.linspace(0.001,0.999,100)
    c=[]
    for z in cvals:
        phiold = np.zeros((10,10))
        err = 1
        st = 1
        ss = st*z
        n, dx = 10, 1
        c.append(ss/st)
        while err >= 1E-6:
            L = build_diffusion_matrix(st, ss, n, dx)
            b = np.ones(n)
            
            phi = np.linalg.solve(L, b)
            err = np.max(np.abs(phi-phiold))
            print err
            phiold = 1*phi
        
        x = np.linspace(0, n*dx, n)
        plt.plot(x, phi)
        
        # Precondition
        P_dsa = scipy.sparse.linalg.LinearOperator(shape=(n,n), \
                matvec=lambda v:  wg_dsa(v, st, ss, n, dx), \
                dtype='f')
    
    
#plt.figure
#params = {'mathtext.default': 'regular' }          
#plt.figure(figsize = (15,10.5))
#plt.rcParams.update(params)
#
#plt.xlabel("Sigma_s/Sigma_t",  fontname="Arial", fontsize=30)
#plt.ylabel("Number of Iterations",  fontname="Arial", fontsize=30)
##plt.xscale('log')
#plt.tick_params(which='major', length=15, labelsize=25)
#plt.tick_params(which='minor', length=7)
##grid(b=True, which='major', color='light grey', linestyle='-')
#plt.grid(True, which='minor', color='lightgrey', linestyle='-')
#plt.grid(True, which='major', color='dimgrey', linestyle='-')
#
#plt.title ("Iterations vs. C",fontsize=30)
#plt.rc('font',family='Arial')
#
##p1 = plt.plot(c, itt, 'b-', label = 'Richardson', linewidth = 4)
#p2 = plt.plot(c, it, 'b-', label = 'GMRES', linewidth = 4)
#
#plt.legend(loc=1,prop={'size':20})
#plt.show()