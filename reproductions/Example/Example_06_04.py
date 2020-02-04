from __future__ import print_function
from robustcontrol import pole_zero_directions, BoundKS, tf, mimotf
from reporting import display_export_data

s = tf([1, 0])

G11 = (s + 2.5) / (s - 2)
G12 = -(0.1 * s + 1) / (s - 2)
G21 = (s + 2.5) / (0.1 * s + 1)
G22 = 1
G = mimotf([[G11, G12],
            [G21, G22]])

p = G.poles()
z = G.zeros()
print('Poles: {0}'.format(p))
print('Zeros: {0}\n'.format(z))


# stable matrix
G11 = (s + 2.5) / (s + 2)
G12 = -(0.1 * s + 1) / (s + 2)
G21 = (s + 2.5) / (0.1 * s + 1)
G22 = 1
Gs = mimotf([[G11, G12],
             [G21, G22]])

# select RHP-pole
p = [2.]
pdir = pole_zero_directions(G, p, 'p')
display_export_data(pdir, 'Poles', ['   u', '   y', '   e '])

up = pdir[0][1]
print('||KS||inf >= {:.3}'.format(BoundKS(Gs, p, up)))
