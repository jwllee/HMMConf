import numpy as np
from hmmconf import numba_utils
from scipy.special import logsumexp
import pytest, timeit, os
from numba import njit


def s2ms(s):
    return s * 1000


def do_test(name, test, setup, repeat, number):
    res = timeit.Timer(test, setup=setup).repeat(repeat, number)
    min_res = s2ms(min(res))
    return min_res


def test_logsumexp1d():
    for n in range(1, 100):
        a = np.random.rand(n)

        print('Testing array with {shape} elements'.format(shape=a.shape[0]))
        print('Array ndim: {}'.format(a.ndim))

        scipy_res = logsumexp(a, keepdims=True)
        numba_res = numba_utils.logsumexp1d(a)

        print('scipy: {}'.format(scipy_res))
        print('numba: {}'.format(numba_res))

        np.testing.assert_almost_equal(numba_res, scipy_res)

    # test zero array
    a = np.full(10, fill_value=-np.inf)
    print('Testing zero array edge case')
    scipy_res = logsumexp(a, keepdims=True)
    numba_res = numba_utils.logsumexp1d(a)

    assert np.isinf(numba_res).all() == True
    np.testing.assert_almost_equal(numba_res, scipy_res)


@pytest.mark.slow
def test_time_logsumexp1d():
    fp = os.path.join('hmmconf', 'numba_utils.py')
    with open(fp) as f:
        setup = f.read()

    setup += '\nfrom scipy.special import logsumexp as scipy_logsumexp'
    setup += '\narr = np.random.rand({})'

    repeat = 3
    number = 10

    scipy_test = 'scipy_logsumexp(arr)'
    numba_test = 'logsumexp1d(arr)'

    for n in range(1, 10):
        print('N: {}'.format(n))
        setup_i = setup.format(n)

        scipy_res = do_test('scipy', scipy_test, setup_i, repeat, number)
        numba_res = do_test('numba', numba_test, setup_i, repeat, number)

        print('scipy: {:.3f}ms'.format(scipy_res))
        print('numba: {:.3f}ms'.format(numba_res))

        # should take longer with scipy
        assert scipy_res > numba_res


def test_logsumexp2d():
    test_msg = 'Testing array with {shape} elements'

    for n_rows in range(1, 10):
        for n_cols in range(1, 10):
            for axis in range(2):
                arr = np.random.rand(n_rows, n_cols)

                print(test_msg.format(shape=arr.shape))

                scipy_res = logsumexp(arr, axis=axis)
                numba_res = numba_utils.logsumexp2d(arr, axis=axis)

                print('scipy: {}'.format(scipy_res))
                print('numba: {}'.format(numba_res))

                np.testing.assert_almost_equal(numba_res, scipy_res)


@pytest.mark.slow
def test_time_logsumexp2d():
    fp = os.path.join('hmmconf', 'numba_utils.py')
    with open(fp) as f:
        setup = f.read()

    setup += '\nfrom scipy.special import logsumexp as scipy_logsumexp'
    setup += '\narr = np.random.rand({}, {})'

    repeat = 3
    number = 10

    scipy_test = 'scipy_logsumexp(arr, {})'
    numba_test = 'logsumexp2d(arr, {})'

    for n_rows in range(1, 5):
        for n_cols in range(1, 5):
            for axis in range(2):
                print('shape: ({} x {}), axis: {}'.format(n_rows, n_cols, axis))
                setup_i = setup.format(n_rows, n_cols)

                scipy_test_i = scipy_test.format(axis)
                numba_test_i = numba_test.format(axis)

                scipy_res = do_test('scipy', scipy_test_i, setup_i, repeat, number)
                numba_res = do_test('numba', numba_test_i, setup_i, repeat, number)

                print('scipy: {:.3f}ms'.format(scipy_res))
                print('numba: {:.3f}ms'.format(numba_res))

                # should take longer with scipy
                assert scipy_res > numba_res


def test_log_normalize1d():
    n = 10

    # inplace
    arr = np.random.rand(n)
    arr = np.log(arr)
    numba_utils.log_normalize1d(arr, inplace=True)
    sum_ = numba_utils.logsumexp1d(arr)[0]
    assert np.isclose(sum_, np.log(1))

    # not inplace
    arr = np.random.rand(n)
    arr = np.log(arr)
    numba_utils.log_normalize1d(arr, inplace=False)
    sum_ = numba_utils.logsumexp1d(arr)[0]
    assert not np.isclose(sum_, np.log(1))

    arr1 = numba_utils.log_normalize1d(arr, inplace=False)
    sum_ = numba_utils.logsumexp1d(arr1)[0]
    assert np.isclose(sum_, np.log(1))


def test_isclose():
    a = 0
    b = 1e-8
    tol = 1e-7
    assert numba_utils.isclose(a, b, tol) == True
    assert numba_utils.isclose(b, a, tol) == True

    a = 0
    b = 1e-7
    tol = 1e-8
    assert numba_utils.isclose(a, b, tol) == False
    assert numba_utils.isclose(b, a, tol) == False

    a = 1e10
    b = 1e10 + 1e-8
    tol = 1e-7
    assert numba_utils.isclose(a, b, tol) == True
    assert numba_utils.isclose(b, a, tol) == True


@njit
def sum_nb(arr):
    res = np.array([0])
    for v in arr:
        res[0] += v
    return res


def test_apply_along_axis_with_sum():
    arr = np.array([[0, 1, 2], [4, 5, 6], [7, 8, 9], [1, 1, 1]])

    res0 = numba_utils.apply_along_axis(sum_nb, 0, arr)
    assert (res0 == [12, 15, 18]).all()

    res1 = numba_utils.apply_along_axis(sum_nb, 1, arr)
    assert (res1 == [3, 15, 24, 3]).all()
