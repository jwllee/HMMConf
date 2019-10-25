import random
import numpy as np
from .utils import *


from hmmconf.base import *
from hmmconf.utils import *
from hmmconf.numba_utils import *


def test_logemissionprob():
    n_obs, n_states = 5, 10
    emitmat = generate_emitmat(n_obs, n_states)
    emitmat_d = generate_emitmat(n_obs, n_states)

    # check that emission probability matrices are valid
    for i in range(n_states):
        sum_ = emitmat[i,:].sum()
        assert np.isclose(sum_, 1)
        sum_ = emitmat_d[i,:].sum()
        assert np.isclose(sum_, 1)

    conf = random.uniform(0, 1)
    obs = 1
    logemitmat = np.log(emitmat)
    logemitmat_d = np.log(emitmat_d)

    logemitprob = compute_logemissionprob(obs, conf, 
                                  logemitmat, logemitmat_d)

    print('Log emission probability: {}'.format(logemitprob))

    emitprob = np.exp(logemitprob)
    assert (emitprob >= 0.).all() and (emitprob <= 1.).all()
    expected = (n_states,)
    assert_shape('log emission prob', expected, logemitprob.shape)


def test_logstateprob():
    n_obs, n_states = 5, 10
    transcube = generate_transcube(n_obs, n_states)
    transcube_d = generate_transcube(n_obs, n_states)

    obs = 1
    conf = random.uniform(0, 1)
    logtranscube = np.log(transcube)
    logtranscube_d = np.log(transcube_d)

    logstateprob = compute_logstateprob(obs, conf, logtranscube, logtranscube_d)
    print('Log state probability: {}'.format(logstateprob))

    stateprob = np.exp(logstateprob)
    assert (stateprob >= 0.).all() and (stateprob <= 1.).all()
    expected = (n_states, n_states)
    assert_shape('State probability', expected, stateprob.shape)

    for state in range(n_states):
        logstateprob_i = logstateprob[state,:]
        sum_ = np.exp(logsumexp1d(logstateprob_i))
        print('Sum of state probability {}: {}'.format(state, sum_))
        assert np.isclose(sum_, 1)

    
