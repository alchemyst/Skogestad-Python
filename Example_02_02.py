import matplotlib.pyplot as plt
import numpy as np
from utils import feedback, tf

#Process model of G with various Controller Gains
s = tf([1, 0], 1)
G = 3*(-2*s + 1) / ((10*s + 1)*(5*s + 1))

tspan = np.linspace(0, 50, 100)

#Calculating the time domian response
for K in [0.5, 1.5, 2, 2.5]:
    T = feedback(G * K, 1)
    [t, y] = T.step(0, tspan)
    plt.plot(t, y)

#Time response plot - Figure 2.6
plt.legend(["Kc = 0.5","Kc = 1","Kc = 2","Kc = 2.5"], 
           bbox_to_anchor=(0, 1.01, 1, 0), loc=3, ncol=4, mode="expand")           
plt.xlabel('Time [s]')
plt.show()
