from __future__ import print_function
import numpy as np

# Define functions
G_3_29 = np.matrix([[5, 4], [3, 2]])
G_3_30 = np.matrix([[0, 100], [0, 0]])


def norms(G):
    """Function to calculate norms"""
    spectrad = max(np.linalg.eigvals(G))
    Frob_norm = np.linalg.norm(G, 'fro')
    B = np.cumsum(abs(G))
    Sum_norm = B[0, G.size - 1]
    Columnsum = np.max(np.sum(abs(G), axis=0))
    Rowsum = np.max(np.sum(abs(G), axis=1))
    [U, S, V] = np.linalg.svd(G)
    Max_singularval = max(S)

    print('Spectral radius = %s' % spectrad)
    print('Frobenius norm = %s' % Frob_norm)
    print('Sum norm = %s' % Sum_norm)
    print('Maximum column sum = %s' % Columnsum)
    print('Maximum row sum = %s' % Rowsum)
    print('Maximum singular value = %s \n' % Max_singularval)

print('Matrix 3.29')
norms(G_3_29)
print('Matrix 3.30')
norms(G_3_30)

