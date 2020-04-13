import matplotlib.pyplot as plt

from utilsplot import mimo_bode, condtn_nm_plot
from utils import tf, mimotf

s= tf([1, 0])

s1 = 1 / (75 * s + 1)
G1 = mimotf([[s1 * 87.8, s1 * -86.4], 
             [s1 * 108.2, s1 * -109.6]])

s2 = 1/(s**2 + 10**2)
G2 = mimotf([[s2 * (s - 1e2), s2 * (10 * (s + 1))], 
             [s2 * (-10 * (s + 1)), s2 * (s - 1e2)]])

processes = [[G1, '(a) Distillation process', -4, 1],
             [G2, '(b) Spinning satellite', -2, 2]]

plt.figure('Figure 3.7')
for i, [G, title, minw, maxw] in enumerate(processes):
    plt.subplot(2, 2, i + 1)
    plt.title(title)
    mimo_bode(G, minw, maxw)
    
    #this is an additional plot to the textbook
    plt.subplot(2, 2, 3 + i)
    condtn_nm_plot(G, minw, maxw)

plt.show()
