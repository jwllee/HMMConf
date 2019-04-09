import numpy as np


if __name__ == '__main__':
    m = np.ones((5, 5))
    l = np.ones((1, 5))
    p = np.ones(5)

    l[0,1] = 2
    l[0,2] = 3
    l[0,3] = 4
    l[0,4] = 5

    print(m)
    print(m + l)
    print(m + l[:,np.newaxis])
    print(p[:,np.newaxis])
