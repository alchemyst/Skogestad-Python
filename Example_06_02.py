import numpy as np

from utils import pole_zero_directions, BoundST, tf, mimotf
from reporting import display_export_data


def G(s):
    p = 3
    z = 2
    d = 30. / 180. * np.pi
    g1 = np.matrix([[1 / (s - p), 0],
                    [0, 1/ (s + 3)]])
    g2 = np.matrix([[np.cos(d), -np.sin(d)],
                    [np.sin(d), np.cos(d)]])
    g3 = np.matrix([[(s - z) / (0.1 * s + 1), 0],
                    [0, (s + 2) / (0.1 * s + 1)]])
    return g1 * g2 * g3 # mimotf scalar multiplication not avaiable yet
    
s = tf([1, 0])
G = mimotf(G(s))
    
p = G.poles()
z = G.zeros()
print 'Poles: {0}'.format(p)
print 'Zeros: {0}'.format(z)
print ''

# selected p & z
p = [3.]
z = [2.]

pdata = pole_zero_directions(G, p, 'p')
zdata = pole_zero_directions(G, z, 'z')
display_export_data(pdata, 'Pole', ['u', 'y'])
display_export_data(zdata, 'Zero', ['u', 'y'])

print 'M_S,min = M_T,min = {:.2f}'.format(BoundST(G, p, z))

print '\nPeak example for deadtime:'
deadtime = np.matrix([[-1, 0],
                      [-2. -3]])
print 'M_T,min = {:.2f}'.format(BoundST(G, p, z), deadtime)
