from utils import bode, tf

s = tf([1, 0], 1)
G = 30 * (s + 1) / ((s + 0.01)**2 * (s + 10))

bode(G, -3, 3, 'Figure 2.3')