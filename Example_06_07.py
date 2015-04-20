import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from utilsplot import dis_rejctn_plot, input_perfect_const_plot, input_acceptable_const_plot
from utils import zeros, distRHPZ


def G(s):
    return 1 / (s + 2) * np.matrix([[s - 1, 4],
                                    [4.5, 2 * (s - 1)]])
                                    
    
def Gdk(s, k):
    return 6 / (s + 2) * np.matrix([[k],
                                    [1]])

def Gd(s):
    k = 1
    return Gdk(s, k)   

def Gdz(s):
    k = sp.Symbol('k')
    return Gdk(s, k)  
    
#z_vec = zeros(G)
#for z in z_vec:
#    eq = distRHPZ(G, Gdz, z)
#    print 'For zero {0} the general condition is {1} < 1'.format(z, eq) # for the solution in textbook, the signs are switched around
#    print 'For k=1, |yzH.gd({0})| is {1}'.format(z, distRHPZ(G, Gd, z))
#    
#    k_range = sp.solve(eq - 1)
#    print 'For acceptable control, k should be in the range {0}.'.format(k_range)
#    print 'The plant is not input-output controllable if k < {0} or k > {1}.'.format(k_range[0], k_range[1])


# The section below demonstrates more utilsplot functions
    
plt.figure('Disturbance rejection example')
dis_rejctn_plot(G, Gd)

plt.figure('Input constraints for perfect control example')
input_perfect_const_plot(G, Gd)

plt.figure('Input constraints for acceptable control example')
input_acceptable_const_plot(G, Gd)
    
plt.show()
