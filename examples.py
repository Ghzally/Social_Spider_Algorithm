import numpy as np


bounds = np.array([1,4,5,6,7])
arr = np.array([100,4,1,3,1])
print(str(bounds*arr))
print(str(np.inner(bounds,arr)))
