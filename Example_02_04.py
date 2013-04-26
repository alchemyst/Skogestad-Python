import matplotlib.pyplot as plt
import numpy as np
from utils import feedback, tf

# Process model of G with various Controller Gains
s=1j
G = 3*(-2*s + 1) / ((10*s + 1)*(5*s + 1))

#Ziegeler and Nicholas tuning rule
#Part of Excercise 2.1
wu = np.angle(G,deg=True)
print wu

#Tu = 1/np.abs(G)
#Pu = 2*np.pi/wu

#Kc = Ku/2.2
#tauI = Pu/1.2
#K = Kc*(1+(1 + 1/(tauI*s))

#T = feedback(G * K, 1)
#[t, y] = T.step(0, tspan)
#plt.plot(t, y, label="") 



#TODO plot y and u(inverse)

#TODO bode plot  T, S, L

#TODO GM, PM

#TODO bandwidth frequency wb, wc, wbt