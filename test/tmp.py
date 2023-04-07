import numpy as np

a = np.ones([2, 3, 3])
b = np.reshape(a, (-1))
print(b)
