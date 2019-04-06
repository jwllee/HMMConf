import numpy as np


if __name__ == '__main__':
    l = np.ones(3)
    n = np.zeros(3)

    with np.errstate(under='ignore'):
        print(l / n)
