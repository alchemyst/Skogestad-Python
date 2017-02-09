import numpy as np
import matplotlib.pyplot as plt

w = 0.2
s = w*1j
G = 5*np.exp(-2*s)/(10*s + 1)

t = np.linspace(0, 100, 200)
# Calculate System Gain and Phase Shift
SG = np.abs(G)
PS = np.angle(G)

u = np.sin(w*t)
y = SG*np.sin(w*t + PS)

plt.figure('Figure 2.1')
plt.title('Sinusoidal response for system G(s)')
plt.plot(t, u)
plt.plot(t, y)
plt.xlabel('Time [s]')
plt.ylabel('Input pertubation, Output response')
plt.legend(['u(t)', 'y(t)'])
plt.title("System response to a sinusodial input")
plt.show()
