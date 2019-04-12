import numpy as np
import utils


def conform(stateprob, obs, confmat, n_states):
    """Computes the conformance of an observation with respect to a state estimation.

    :param stateprob array_like: state estimation vector that sums to 1.
    :param obs int: observation
    """
    utils.assert_shape('stateprob', (1, n_states), stateprob.shape)

    if not np.isclose(stateprob.sum(), [1.]):
        raise ValueError('State estimation: {} does not sum to 1.'.format(stateprob))

    v = np.dot(stateprob, confmat[obs])
    utils.assert_no_negatives('conformance', v)

    # handle floating point imprecision that can make conformance go over 1.
    if v[0] > 1. and v[0] - 1. < 1e-10:
        v[0] = 1.

    utils.assert_bounded('conformance', v, 0., 1.)
    return v


