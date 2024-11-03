import numpy as np
from scipy.stats import ortho_group

d = 3
seed = 1
size = 2
a, b = np.float32(ortho_group.rvs(size=size, dim=d, random_state=seed))
print(a)
print(np.linalg.det(a))
print(a @ a.T)