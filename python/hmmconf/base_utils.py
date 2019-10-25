import numpy as np
from collections import deque
import sys


from hmmconf.utils import *


__all__ = [
    'ConvergenceMonitor',
    'partition_X',
    'assert_transcube_validity',
    'assert_emitmat_validity',
    'STATS_NOBS',
    'STATS_STARTPROB',
    'STATS_C_TRANS_LOG_NUMERATOR',
    'STATS_C_OBS_NUMERATOR',
    'STATS_NC_TRANS_LOG_NUMERATOR',
    'STATS_NC_OBS_NUMERATOR',
    'N_PARAMS',
    'PARAM_START',
    'PARAM_CONFORM_TRANS',
    'PARAM_CONFORM_OBS',
    'PARAM_NCONFORM_TRANS',
    'PARAM_NCONFORM_OBS'
]


STATS_NOBS = 0
STATS_STARTPROB = 1
STATS_C_TRANS_LOG_NUMERATOR = 2
STATS_NC_TRANS_LOG_NUMERATOR = 3
STATS_C_OBS_NUMERATOR = 4
STATS_NC_OBS_NUMERATOR = 5

N_PARAMS = 5
PARAM_START = 0
PARAM_CONFORM_TRANS = 1
PARAM_CONFORM_OBS = 2
PARAM_NCONFORM_TRANS = 3
PARAM_NCONFORM_OBS = 4


class ConvergenceMonitor:
    """Monitors and reports convergence to :data:`sys.stderr`.

    **Parameters**:
    :param tol float: Convergence tolerence. EM has converged either if the maximum number of iterations is 
                or the log probability improvement between two consecutive iterations is less than tol.
    :param n_iter: Maximum number of iterations to perform.
    :param verbose: If ``True`` then per-iteration convergence reports are printed, otherwise the 
                    monitor is mute.

    **Attributes**:
    :param history deque: The log probability of the data for the last two training iterations. 
    :param iter int: Number of iterations performed while training the model.
    """
    _header = '{:>10s} {:>16s} {:>16s}'.format('iteration', 'logprob', 'delta')
    _template = "{iter:>10d} {logprob:>16.4f} {delta:>+16.4f}"

    def __init__(self, tol, n_iter, verbose):
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.history = deque(maxlen=2)
        self.iter = 0
        self.logger = make_logger(ConvergenceMonitor.__name__)

    def __repr__(self):
        class_name = ConvergenceMonitor.__name__
        params = dict(vars(self), history=list(self.history))
        return "{}({})".format(class_name, _pprint(params, offset=len(class_name)))

    def _reset(self):
        """Resets the monitor.
        """
        self.iter = 0
        self.history.clear()

    def report(self, logprob):
        """Report convergence to :data:`sys.stderr`.

        The output consists of three columns: iteration number, log probability of the data
        at the current iteration and convergence rate. At the first iteration convergence 
        rate is unknown and is denoted as nan.

        :param logprob float: Log likelihood of the data
        """
        if self.verbose:
            delta = logprob - self.history[-1] if self.history else np.nan
            if not self.history:
                print(self._header, file=sys.stderr)
            message = self._template.format(iter=self.iter + 1, logprob=logprob, delta=delta)
            print(message, file=sys.stderr)

        if len(self.history) > 0 and self.history[-1] > logprob:
            msg = 'Log probability is NOT non-decreasing from previous {:.2f} to current {:.2f}'
            # raise ValueError(msg.format(self.history[-1], logprob))

        self.history.append(logprob)
        self.iter += 1

    @property
    def converged(self):
        """``True`` if the EM algorithm converged and ``False`` otherwise.
        """
        converged_ = (self.iter == self.n_iter or 
                     (len(self.history) == 2 and
                     abs(self.history[1] - self.history[0]) < self.tol))
        return converged_


def partition_X(X, lengths, n):
    # do not partition if it's less than 2n
    if lengths.shape[0] < 2 * n:
        # msg = 'Using 1 processor rather than {} since there is only {} sequences'
        # msg = msg.format(self.n_jobs, lengths.shape[0])
        # self.logger.debug(msg)
        return [X,], [lengths,]

    # workout the args
    n_samples = X.shape[0]
    end = np.cumsum(lengths).astype(np.int32)
    if end[-1] > n_samples:
        msg = 'More than {:d} samples in lengths array {}'
        msg = msg.format(n_samples, lengths)
        raise ValueError(msg)

    end_list = np.array_split(end, n, axis=0)
    split_inds = list(map(lambda a: a[-1], end_list))
    # leave out the last end which should equal len(X)
    split_inds = split_inds[:-1] 
    split_inds = np.asarray(split_inds)

    X_parts = np.split(X, split_inds)
    lengths_parts = np.array_split(lengths, n)

    # utils.assert_shape('X_parts', (n,), (len(X_parts),))
    # utils.assert_shape('lengths_parts', (n,), (len(lengths_parts),))

    return X_parts, lengths_parts


def assert_transcube_validity(transcube, n_obs, n_states):
    for o in range(n_obs):
        transmat = transcube[o,:,:]
        assert_shape('transmat_d', (n_states, n_states), transmat.shape)
        sumframe = transmat.sum(axis=1)
        almost_one = np.isclose(sumframe, np.ones(n_states))
        error_rows = sumframe[np.invert(almost_one)]
        errmsg = 'transmat_d[{},:,:] does not sum to almost 1: {}'.format(o, error_rows)
        assert almost_one.all(), errmsg


def assert_emitmat_validity(emitmat, n_obs, n_states):
    sumframe = emitmat.sum(axis=1)
    almost_one = np.isclose(sumframe, np.ones(n_states))
    error_rows = sumframe[np.invert(almost_one)]
    errmsg = 'emitmat_d does not sum to 1 to almost 1: {}'.format(error_rows)
    assert almost_one.all(), errmsg
