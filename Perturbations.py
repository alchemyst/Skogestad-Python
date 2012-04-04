import numpy as np
import itertools


def Possibilities(umin, umax, Amount):
    """this function is to calculate all possible perturbations of a maximum and minimum of a vector of parameters"""

    def box_ready(umin, umax, Amount):
        """create a suitable matrix for the mesh function"""
        umin, umax = np.matrix(umin), np.matrix(umax)
        box = np.zeros((len(umin), 3))
        for i in range(2):
            for j in range(len(umin)):
                if i ==0:
                    box[j, i] = umin[j]
                if i ==1:
                    box[j, i] = umax[j]
        box[:, -1] = Amount
        return box

    def coords(box):
        return [entry[:2] for entry in box]

    def internalmesh(box):
        """ generate internal points linearly spaced inside of a box """
        return itertools.product(*[np.linspace(*b) for b in box])

    def surfmesh_slow(box):
        """ Generate points on the edges of a box by generating an
        internal grid and then selecting the points on the outside"""
        return (point for point in internalmesh(box)
                if any(p in minmax for p, minmax in zip(point, coords(box))))

    def surfmesh(box):
        """ generate points on the edges of a box """
        Ndims = len(box)
        dimindex = range(Ndims)
        # have at least one constrained dimension
        for Nunconstrained in range(0, Ndims):
            # select all ways of generating this 
            for dims in itertools.combinations(dimindex, Nunconstrained):
                pointbase = coords(box)
                for i in dims:
                    pointbase[i] = np.linspace(*box[i])[1:-1]
                for coord in itertools.product(*pointbase):
                    yield coord


    box=box_ready(umin, umax, 2)

    vec=[]
    for c in surfmesh(box):
        vec.append(c)

    perturbations =np.matrix(vec)

    return perturbations


#for example a matrix of the minimum and maximum values of a certain set of parameters
umin     =[[85], [160], [0.1], [1]]
umax     =[[95], [120], [0.8], [4]]

#create a matrix of all possibilities of the umin and umax vectors
#the first entry of the matrix correspondse to the first entry in the minimum and maximum matrices
Possible = Possibilities(umin, umax, 2)
print Possible
print Possible.shape[0]
