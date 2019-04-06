import os, sys

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)

sys.path.insert(0, parentdir)

import utils
import numpy as np


if __name__ == '__main__':
    cube = np.zeros((3, 4, 3))
    cube[0, 1, 1] = 1
    cube[1, :, :] = 2
    cube[2, :, :] = 3

    print(cube)
    utils.normalize(cube, axis=2)
    print(cube)

    print((cube > 0).astype(int))
