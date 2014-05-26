import numpy as np

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
        [U, s, V] = np.linalg.svd(M)
        S = np.zeros((2,2))
        S[0,0] = s[0]
        S[1,1] = s[1]
        delta = 1/S[0,0] * V[:,0] * U[:,0].H  
    elif (DeltaStructure == 'Diagonal'):
# TODO: complete
        delta = 'NA'
    else: delta = 0
    return delta

    

