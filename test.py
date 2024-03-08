
import numpy as np
import scipy
import time
arr = np.array([[ 0.24475796, -0.06991703],
 [-0.12180452, -0.05139566],
 [ 0.1369748,  0.15667394],
 [ 0.28,        0.28      ]])
pos = [1, 2, 3]
print(pos[0:2])
print(arr.shape)
t_s = time.time()
for i in range(100):
    hull = scipy.spatial.ConvexHull(arr)
    #print(hull.volume)
t_end = time.time()
print(t_end - t_s)
