import numpy as np

from hmmconf.utils import *
from hmmconf.numba_utils import *


__all__ = [
    'generate_log_prob_vector',
    'generate_emitmat',
    'generate_confmat',
    'generate_transcube'
]


def generate_log_prob_vector(n):
    arr = np.random.rand(n)
    log_normalize(arr)
    return arr


def generate_emitmat(n_obs, n_states):
    # has to be of dimension (n_states, n_obs)
    emitmat = np.random.rand(n_states, n_obs)
    # make some zero
    for row in range(n_states):
        n_zeros = int(n_obs/ 2)
        ids = np.random.choice(n_obs, n_zeros)
        emitmat[row,ids] = 0
    # normalize to become valid probability
    normalize(emitmat, axis=1)
    return emitmat


def generate_confmat(n_obs, n_states):
    emitmat = generate_emitmat(n_obs, n_states)
    confmat = (emitmat != 0).astype(float)
    return confmat


def generate_transcube(n_obs, n_states):
    transcube = np.random.rand(n_obs, n_states, n_states)
    normalize(transcube, axis=2)
    # check that transcube is valid
    assert np.isclose(transcube[0, 0, :].sum(), 1)
    return transcube
