import numpy as np
from pytest import fixture

from .utils import *

from hmmconf.utils import *
from hmmconf.conform import *


def test_conform_computes():
    n_obs, n_states = 5, 10
    logstateprob = generate_log_prob_vector(n_states)
    print('log state probability: {}'.format(logstateprob))

    confmat = generate_confmat(n_obs, n_states)
    print('conform mat: {}'.format(confmat))

    with np.errstate(under='ignore'):
        logconfmat = np.log(confmat)

    obs = 1
    res = logconform(logstateprob, obs, confmat)
    print('Conformance value: {}'.format(res))

    assert res >= 0 and res <= 1
