import numpy as np

a = np.random.randint(low=0, high=20000, size=(500000, 2))

%timeit np.unique(a, axis=0)

>>> 500 ms
