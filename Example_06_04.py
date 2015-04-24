from utils import pole_zero_directions, BoundKS, tf, mimotf
from reporting import display_export_data

s = tf([1, 0])

G11 = (s - 2.5) / (s - 2)
G12 = -(0.1 * s + 1) / (s + 2)
G21 = (s - 2.5) / (0.1 * s + 1)
# TODO ISSUE the mimotf does not accept constants in a cell
G22 = 1

G = mimotf([[G11, G12],
            [G21, G22]])

p = G.poles()
z = G.zeros()
print 'Poles: {0}'.format(p)
print 'Zeros: {0}'.format(z)
print ''
    
G11 = (s + 2.5) / (s + 2)
G12 = -(0.1 * s + 1) / (s + 2)
G21 = (s + 2.5) / (0.1 * s + 1)   
G22 = 1    
Gs = mimotf([[G11, G12],
             [G21, G22]])
    
# select RHP-pole
p = [2.]
# TODO input direction differ from textbook
pdir = pole_zero_directions(Gs, p, 'p')
display_export_data(pdir, 'Poles', ['u', 'y'])

print '||KS|| > {:.3}'.format(BoundKS(Gs, p))
