# -*- coding: utf-8 -*-
# Example 3.5: Distillation process

import numpy as np

G = np.matrix([[87.8, -86.4],[108.2, -109.6]]);

[U,S,V] = np.linalg.svd(G)
conditionNum = np.linalg.cond(G)

# The matrices U and V are different from the text. Refer to Skogestad, 2nd edition, page 78 for the reasoning.

print 'U = ',U
print ''
print 'Sigma = ', S
print ''
print 'V = ', V
print ''
print 'The condition number is: ',conditionNum