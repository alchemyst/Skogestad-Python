import numpy as np
import itertools


def possibilities(umin, umax, Amount):
    """This function is to calculate all possible perturbations of a maximum and
    minimum of a vector of parameters"""

    def box_ready(umin, umax, Amount):
        """Create a suitable matrix for the mesh function"""
        return zip(umin, umax, [Amount]*len(umin))

    def coords(box):
        return [entry[:2] for entry in box]

    def internalmesh(box):
        """Generate internal points linearly spaced inside of a box"""
        return itertools.product(*[np.linspace(*b) for b in box])

    def surfmesh_slow(box):
        """Generate points on the edges of a box by generating an internal
        grid and then selecting the points on the outside"""
        return (point for point in internalmesh(box)
                if any(p in minmax for p, minmax in zip(point, coords(box))))

    def surfmesh(box):
        """Generate points on the edges of a box"""
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

    box = box_ready(umin, umax, Amount)
    return np.array(list(surfmesh(box)))

if __name__ == '__main__':

    #for example a matrix of the minimum and maximum values of a certain set of parameters
    umin = [0, 0, 10]
    umax = [1, 1, 20]

    # Create a matrix of all possibilities of the umin and umax vectors
    # The first entry of the matrix correspondse to the first entry in
    # The minimum and maximum matrices
    possible = possibilities(umin, umax, 3)

    print possible
    print possible.shape[0]

    umin = np.ones(20)
    umax = 2*np.random.random(20)
    print umax
    print umin

    print possibilities(umin, umax, 1)
