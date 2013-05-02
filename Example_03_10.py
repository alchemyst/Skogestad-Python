import numpy as np
from utils import RGA

# 3x3 plant at steady-state
G = np.matrix([[16.8, 30.5, 4.30], 
			   [-16.7, 31.0, -1.41], 
			   [1.27, 54.1, 5.40]])

print RGA(G)

# pairing rule 2: avoid pairing negative RGA elements
# RGA = [[1.50, 0.99, -1.48], [-0.41, 0.97, 0.45], [-0.08, -0.95, 2.03]]
# Therefore pair the diagonal elements because they are positive, this means
# use u1 to control y1, u2 to control y2 and u3 to control y3
