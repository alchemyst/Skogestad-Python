import numpy as np
import matplotlib.pyplot as plt

w = 0.2 #rad/s
s = w*1j
G = 5*np.exp(-2*s)/(10*s + 1)

t = np.linspace(0, 100, 200)
GN = np.abs(G)      #Gain at w = 0.2 rad/s
PS = np.angle(G)    #Phase Shift at w = 0.2 rad/s

u = np.sin(w*t)
y = GN*np.sin(w*t + PS)

plt.figure('Figure 2.1')
plt.title('Sinusoidal response for system G(s)')
plt.plot(t, u)
plt.plot(t, y)
plt.xlabel('Time [s]')
plt.ylabel('Input pertubation, Output response')
plt.legend(['u(t)', 'y(t)'])
plt.title("System response to a sinusodial input")
plt.show()
