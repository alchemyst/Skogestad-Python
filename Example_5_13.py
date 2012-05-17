
"This code plots figure 5.16 of Example 5.13"

import numpy as np
import matplotlib.pyplot as plt

w = np.logspace(-3, 3, 1000)
s = w*1j

def G(s):    
    G = 5/((10*s+1)*(s+1))
    return G

def Gd(s):    
    Gd = 0.54/((s+1)*(0.2*s+1))
    return Gd

#Plotting figure 5.16 (a)
plt.figure(1)
plt.loglog(w, [np.abs(G(thisw)) for thisw in w])
plt.loglog(w, [np.abs(Gd(thisw)) for thisw in w])
plt.loglog(w, np.ones(len(w)), '-.')
plt.title('|G| & |Gd| over frequency')
plt.xlabel('frequency [rad/s]')
plt.ylabel('Magnitude') 
plt.legend(('|G|','|Gd|'), loc=1)

plt.show()    



