from __future__ import print_function

from utils import tf, mimotf

s = tf([1, 0], 1)
G = 1/(1.25*(s + 1)*(s + 2)) * mimotf([[s - 1, s],
                                       [-6, s - 2]])

print('Poles: ', G.poles())
print('Zeros: ', G.zeros())
