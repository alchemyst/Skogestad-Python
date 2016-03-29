from __future__ import print_function
import numpy as np

from robusttools import UnstructuredDelta


M = np.matrix([[2., 2.],[-1., -1.]])

print('Full perturbation delta matrix')
print(UnstructuredDelta(M, 'Full'),'\n')
print('Diagonal perturbation delta matrix')
print(UnstructuredDelta(M, 'Diagonal'))
