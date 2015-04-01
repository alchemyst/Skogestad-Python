import matplotlib.pyplot as plt
import numpy as np

from utilsplot import dis_rejctn_plot, input_perfect_const_plot, input_acceptable_const_plot

def G(s):
    return 1 / (s + 2) * np.matrix([[s + 1, 4],
                                    [4.5, 2 * (s - 1)]])
    
def Gdk(s, k):
    return 6 / (s + 2) * np.matrix([[k],
                                    [1]])

# TODO create yzH.gd(z) function

# The section below demonstrates more utilsplot functions
def Gd(s):
    k = 1
    return Gdk(s, k)    
    
plt.figure('Disturbance rejection example')
dis_rejctn_plot(G, Gd)

plt.figure('Input constraints for perfect control example')
input_perfect_const_plot(G, Gd)

plt.figure('Input constraints for acceptable control example')
input_acceptable_const_plot(G, Gd)
    
plt.show()
