

"This code plots figure 5.24 of Exercise 5.12"

import numpy as np
import matplotlib.pyplot as plt

w = np.logspace(-4, -1, 1000)
s = w*1j

for n in [1, 2, 3, 4]:
    tau_h = 1000
    h_n = 1/((tau_h/n)*s+1)**n
    plt.loglog(w*tau_h, abs(h_n))
    plt.title('|h_n for different values of n|')
    plt.xlabel('frequency [rad/s]')
    plt.ylabel('Magnitude') 
    plt.legend(('n=1', 'n=2', 'n=3', 'n=4'), loc=6) 
        
plt.show()    

