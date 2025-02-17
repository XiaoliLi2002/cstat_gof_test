import numpy as np

Y=np.zeros((3,3))+np.eye(3)-np.asmatrix(np.array([[1,0,0],[1,0,0],[0,0,1]]))
print(Y)
print(np.mean(Y,axis=0))