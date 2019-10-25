import numpy as np
from numba import njit, prange


__all__ = [
    'logsumexp1d',
    'logsumexp2d',
    'log_normalize1d',
    'isclose'
]


@njit(parallel=True)
def apply_along_axis(func1d, axis, arr):
    """Apply a 1d function on a 2d array along a specified axis.

    :param func1d: 1d function to apply, need to return a 1d array result
    :param axis: axis on which function should be applied on
    :param arr: 2d array of data

    Examples
    ========

    >>> def sum_nb(arr):
    ...     res = np.array([0])
    ...     for v in arr:
    ...         res[0] += v
    ...     return res
    >>> arr = np.array([[0, 1, 2], [4, 5, 6], [7, 8, 9], [1, 1, 1]])
    >>> arr
    array([[0, 1, 2],
           [4, 5, 6],
           [7, 8, 9],
           [1, 1, 1]])
    >>> res0 = apply_along_axis(sum_nb, 0, arr)
    >>> res0
    array([12, 15, 18])

    >>> res1 = apply_along_axis(sum_nb, 1, arr)
    >>> res1
    array([3, 15, 24, 3])
    """
    assert arr.ndim == 2
    assert axis in [0, 1]

    if axis == 0:
        res = np.empty(arr.shape[1])
        for i in prange(len(res)):
            res[i] = func1d(arr[:, i])[0]
    else:
        res = np.empty(arr.shape[0])
        for i in prange(len(res)):
            res[i] = func1d(arr[i, :])[0]

    return res


@njit('f8[:](f8[:])', fastmath=True)
def logsumexp1d(arr):
    assert arr.ndim == 1
    out = np.empty(1)

    # check that it is not a zero array to avoid zero division
    all_zeros = np.isinf(arr).all()
    if all_zeros:
        out[0] = -np.inf
        return out

    max_ = arr.max()
    ds = arr - max_
    sum_ = np.exp(ds).sum()

    out[0] = np.log(sum_) + max_

    return out


@njit('f8[:](f8[:,:], u2)')
def logsumexp2d(arr, axis):
    res = apply_along_axis(logsumexp1d, axis, arr)
    return res


@njit('(f8[:], b1)')
def log_normalize1d(arr, inplace):
    """Assumes that arr does not sum to 0!
    """
    # zero array check
    sum_ = logsumexp1d(arr)[0]
    if np.isinf(sum_):
        raise ValueError

    if inplace:
        arr[:] = arr - sum_
    else:
        return arr - sum_


@njit
def isclose(a, b, tol):
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError
    return np.less_equal(abs(a - b), tol)
