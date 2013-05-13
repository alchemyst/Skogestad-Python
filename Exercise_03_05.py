import numpy as np

# Define functions
G_3_29 = np.matrix([[1, 2], [-3, 4]])
G_3_30 = np.matrix([[0, 100], [0, 0]])

def norms(G):
    """Function to calculate norms"""
    spectrad = max(np.linalg.eigvals(G))
    Frob_norm = np.linalg.norm(G, 'fro')
    B = np.cumsum(abs(G))
    Sum_norm = B[0, G.size-1]
    Columnsum = np.max(sum(abs(G)))
    Rowsum = np.max(sum(abs(G.T)))
    [U, S, V] = np.linalg.svd(G)
    Max_singularval = max(S)

    print 'Spectral radius =', spectrad
    print 'Frobenius norm =', Frob_norm
    print 'Sum norm =', Sum_norm
    print 'Maximum column sum =', Columnsum
    print 'Maximum row sum =', Rowsum
    print 'Maximum singular value =', Max_singularval

norms(G_3_29)
