import matplotlib.pyplot as plt

from control import bode_plot, tf, pade
from numpy import linspace

G = tf([5], [10, 1])        #Define the transfer function
num, den = pade(2, n=10)     #Generate a 10th order pade approximation for 2s delay
P = tf(num, den)            #General transfer function of the delay
Gdelay = G*P                #Apply the delay to the transfer function
plt.figure('Figure 2.2')
bode_plot(Gdelay, omega=linspace(0.001, 10, 1000))