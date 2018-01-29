from __future__ import print_function
import numpy as np

from utils import pole_zero_directions, BoundST, tf, mimotf, RHPonly
from reporting import display_export_data

s = tf([1, 0])

p = 3
z = 2
d = 30. / 180. * np.pi
g1 = mimotf([[1 / (s - p), 0],
             [0, 1 / (s + 3)]])
g2 = mimotf([[np.cos(d), -np.sin(d)],
             [np.sin(d), np.cos(d)]])
g3 = mimotf([[(s - z) / (0.1 * s + 1), 0],
             [0, (s + 2) / (0.1 * s + 1)]])
G = g1 * g2 * g3

p = G.poles()
z = G.zeros()
print('All Poles: {0}'.format(p))
print('All Zeros: {0}\n'.format(z))

RHPzeros = RHPonly(z)
RHPpoles = RHPonly(p)
print("RHP poles only: ", RHPpoles)
print("RHP zeros only: ", RHPzeros)

# selected p & z
p = [3.]
z = [2.]

pdata = pole_zero_directions(G, p, 'p')
zdata = pole_zero_directions(G, z, 'z')
rowhead = ['u','y','e ']
display_export_data(pdata, 'Poles', rowhead)
display_export_data(zdata, 'Zeros', rowhead)

zdata,_ = pole_zero_directions(G, z, 'z', 'y')
print('M_S,min = M_T,min = {:.2f}'.format(BoundST(G, p, z)))

# TODO fix BoundST with deadtime
print('\nPeak example for deadtime:')
deadtime = np.matrix([[-1, 0],
                      [-2., -3]])
     
print('M_T,min = {:.2f}'.format(BoundST(G, p, z, deadtime)))
