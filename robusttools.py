import numpy as np
import utils

'''Use this file to add more MIMO functions for robust stability and performance'''

def UnstructuredDelta(M, DeltaStructure):
    '''    
    Function to calculate the unstructured delta stracture for a pertubation
    
    Parameters
    ----------
    M : matrix (2 by 2), 
        TODO: extend for n by n
        M-delta structure
        
    DeltaStructure : string ['Full','Diagonal']
        Type of delta structure
        
    Returns
    -------    
    delta : matrix
        unstructured delta matrix
    '''
    
    if (DeltaStructure == "Full"):
        [U, s, V] = utils.SVD(M)
        S = np.diag(s)
        delta = 1/s[0] * V[:,0] * U[:,0].H  
    elif (DeltaStructure == 'Diagonal'):
# TODO: complete
        delta = 'NA'
    else: delta = 0
    return delta

    

