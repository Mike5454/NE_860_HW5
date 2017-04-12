# Import Settings and dsa file
import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from dsa import wg_dsa
from scipy.sparse.linalg import LinearOperator as LO

#Define parameters
num_cells = 100
num_angles = 8
dx = 1
# This will get our c ratio to be used in the graphing
cvals = np.linspace(0.001,0.999,100)
c=[]
itt=[]
ittt=[]
# for loop to repeatedly solve the transport equations
for z in cvals:
    Sigma_t = 1.0
    Sigma_s = Sigma_t * z
    c.append(Sigma_s/Sigma_t)
    
# define a sweep operator to aid in solving for the flux
    def sweep_operator(q) :
        # use numpy to get the abscissa and weights; store only positive half-space
        mu, w = np.polynomial.legendre.leggauss(2*num_angles)
        # mu values based on legendre polynomials
        mu = mu[num_angles:]
        # weight values based on mu values
        w  = w[num_angles:]
        phi = np.zeros(num_cells)
        # loop over two half spaces
        for o in range(0, 2) :
            # vacuum conditions assumed
            psi_edge = np.zeros(num_angles)
            # loop over all spatial cells in the correct order for this half space
            i_min = 0; i_max = num_cells; i_inc = 1
            if o :
                i_min = num_cells-1; i_max = -1; i_inc = -1
            for i in range(i_min, i_max, i_inc) :
                a = 2.0 * mu / dx
                b = 1.0 / (Sigma_t + a)
                # flux values at edges and center for each discritized cell
                psi_center = b * (q[i] + a[:] * psi_edge[:])
                psi_edge[:] = 2.0*psi_center[:] - psi_edge[:]            
                phi[i] += np.dot(psi_center, w)
        
        return phi
    
    def transport_operator(phi) :
        # M*S*phi
        MS_times_phi = phi * Sigma_s / 2.0
        # T*M*S
        TMS_times_phi = sweep_operator(MS_times_phi)
        # (I-TMS)phi
        return phi - TMS_times_phi
        
    def residual_print(r, i=0) :
        print("iter=%i   ||Ax-b||=%.4e " %(i, np.linalg.norm(r)))
         
        
# Definition for richardson iterations       
    def richardson(AA, b, P = None, tol = 1e-10, maxiter = 100) :
    
        if P :
            A = LO(shape=(num_cells,num_cells), matvec=lambda v: P(AA(v)))
            b = P(b)
        else :
            A = AA
        # Make and x0 matrix to initiate the calculations    
        x0 = np.zeros(len(b))
        r0 = np.linalg.norm(A.matvec(x0)-b)
        for i in range(0, maxiter) :
            # solve for x which is the updated flux matrix and check the error
            x = x0 - A.matvec(x0) + b
            # If the error is met append i number of iterations
            err = np.max(np.abs(x-x0)) 
            if err < 1e-6 :
                itt.append(i)
                break
            x0 = 1.*x
            r = A.matvec(x0)-b
            #residual_print(r, i)
            if np.linalg.norm(r) < tol or np.linalg.norm(r)/r0 < tol :
                break
        # Make the number of iterations global so it may be called in graphing
        global itt
        # return phi
        return x
# callback function to keep track of GMRES iterations
    def make_callback():
        closure_variables = dict(counter=0, residuals=[]) 
        def callback(residuals):
            closure_variables["counter"] += 1
            closure_variables["residuals"].append(residuals)
            print closure_variables["counter"], residuals
            if residuals < 1*10**-6:
                # if tolerence is met append the iteration count to be used for graphing
                ittt.append(closure_variables["counter"])
        return callback
        
    s = np.ones(num_cells)
    # perform sweep to get vector values
    b = sweep_operator(s/2.0)
    print b
    
    P_dsa = LO(shape=(num_cells,num_cells), \
               matvec=lambda v:  wg_dsa(v, Sigma_t, Sigma_s, num_cells, dx), \
               dtype='f')
    
    # Define linear operator that uses the trasport definition to be used in richarson and gmres
    A = scipy.sparse.linalg.LinearOperator(shape=(num_cells,num_cells), \
                                           matvec=transport_operator, dtype='f')
    
    #print("RICHARDSON")
    #phi = richardson(A, b, P=P_dsa, tol=1e-10,maxiter=1000)
    #print phi
    
#    print("GMRES")
#    phi = scipy.sparse.linalg.gmres(A, b, M=None, tol=1e-6,maxiter=1000,callback=make_callback())
#    print phi


plt.figure
params = {'mathtext.default': 'regular' }          
plt.figure(figsize = (15,10.5))
plt.rcParams.update(params)

plt.xlabel("Sigma_s/Sigma_t",  fontname="Arial", fontsize=30)
plt.ylabel("Number of Iterations",  fontname="Arial", fontsize=30)
#plt.xscale('log')
plt.tick_params(which='major', length=15, labelsize=25)
plt.tick_params(which='minor', length=7)
#grid(b=True, which='major', color='light grey', linestyle='-')
plt.grid(True, which='minor', color='lightgrey', linestyle='-')
plt.grid(True, which='major', color='dimgrey', linestyle='-')

plt.title ("Iterations vs. C",fontsize=30)
plt.rc('font',family='Arial')

#p1 = plt.plot(c, itt, 'b-', label = 'Richardson', linewidth = 4)
#p2 = plt.plot(c, ittt, 'b-', label = 'GMRES', linewidth = 4)

plt.legend(loc=1,prop={'size':20})
plt.show()
