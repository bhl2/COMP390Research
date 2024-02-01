import PyKDL
import numpy as np
axe = 1/np.sqrt(2)
axis = [axe, axe, 0]
rot = np.pi/4

rotMat = PyKDL.Rotation.Rot(axis, rot)
print(rotMat)
#print(dir(PyKDL))