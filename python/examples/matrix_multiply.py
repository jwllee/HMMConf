import numpy as np


if __name__ == '__main__':
    a0 = np.ones((3, 3)) * 2.
    a1 = np.ones((3, 3)) * 3.

    print('Using np.multiply: \n{}'.format(np.multiply(a0, a1)))
    print('Normal multiply: \n{}'.format(a0 * a1))
    print('Matrix multiply: \n{}'.format(np.matmul(a0, a1)))

