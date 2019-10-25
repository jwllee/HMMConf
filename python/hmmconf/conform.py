from numba import njit
import numpy as np
from hmmconf.numba_utils import *
from hmmconf.utils import *


logger = make_logger(__file__)


__all__ = [
    'logconform'
]


@njit('f8(f8[:], i8, f8[:,:])')
def logconform(logmarkingprob, obs, confmat):
    """
    Note that this is specific to Petrinet markings and not Hidden Markov Model states!!
    """
#    assert logmarkingprob.ndim == 1
    # logstateprob.shape == (n_states,)
    # confmat.shape == (n_obs, n_states)
#    assert logmarkingprob.shape[0] == confmat.shape[1]

    n_states = logmarkingprob.shape[0]

    # check that logstateprob is valid, until 1e-08 should be 
    # enough since that's the default in numpy
    sum_ = np.exp(logsumexp1d(logmarkingprob))[0]
    if abs(sum_ - 1) > 1e-8:
        err_msg = 'State probability does not sum to 1'
        print(err_msg)
#        logger.error(err_msg + ': {}'.format(sum_))
        return -1.

    markingprob = np.exp(logmarkingprob)
    obs_confmat = confmat[obs, :] 

    res = 0
    for i in range(n_states):
        res = res + markingprob[i] * obs_confmat[i]

    # res = np.dot(stateprob, obs_confmat)

    # handle possible floating point imprecision
    if res > 1. and res - 1. < 1e-8:
        res = 1.

    return res


def conform(stateprob, obs, confmat, n_states):
    """Computes the conformance of an observation with respect to a state estimation.

    :param stateprob array_like: state estimation vector that sums to 1.
    :param obs int: observation
    """
    # utils.assert_shape('stateprob', (1, n_states), stateprob.shape)

    if not np.isclose(stateprob.sum(), [1.]):
        raise ValueError('State estimation: {} does not sum to 1.'.format(stateprob))

    v = np.dot(stateprob, confmat[obs])
    # utils.assert_no_negatives('conformance', v)

    # handle floating point imprecision that can make conformance go over 1.
    if v[0] > 1. and v[0] - 1. < 1e-10:
        v[0] = 1.

    # utils.assert_bounded('conformance', v, 0., 1.)
    return v
