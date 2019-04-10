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

    print('new experiment...')

    m = np.ones((5, 5, 5))
    m = np.ones((5, 5))
    m[0,:] = 0.
    m[2,:] = 0.

    print(m)

    print('keepdims is False')

    framesum = np.sum(m, axis=1, keepdims=False)
    to_select = np.argwhere(framesum==0.).ravel()
    print(framesum)
    print(to_select)
    print(m[to_select,:])

    print('keepdims is True')

    framesum = np.sum(m, axis=1, keepdims=True)
    to_select = np.argwhere(framesum==0.)
    print(framesum)
    print(to_select)
    print(m[to_select])

    print('selection experiment...')
    t = np.ones((4, 5, 5))
    m = np.zeros((4, 5, 5))
    m[0,:,:] = 1
    m[2,:,:] = 1

    print(m)

    row_sum = m.sum(axis=2)
    row_ind = np.argwhere(row_sum == 0.)
    print(row_ind)
    get0 = lambda a: a[0]
    get1 = lambda a: a[1]
    ind0 = np.apply_along_axis(get0, 1, row_ind)
    ind1 = np.apply_along_axis(get1, 1, row_ind)
    print(ind0)
    print(ind1)
    m[ind0,ind1,:] = t[ind0,ind1,:]

    print(m)
