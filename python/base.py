from collections import deque
from sklearn.base import _pprint
from scipy.misc import logsumexp
import numpy as np
import pandas as pd
import utils, warnings, sys


class ConvergenceMonitor:
    _template = "{iter:>10d} {logprob:>16.4f} {delta:>+16.4f}"

    def __init__(self, tol, n_iter, verbose):
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.history = deque(maxlen=2)
        self.iter = 0

    def __repr__(self):
        class_name = self.__class__.__name__
        params = dict(vars(self), history=list(self.history))
        return "{}({})".format(class_name, _pprint(params, offset=len(class_name)))

    def _reset(self):
        self.iter = 0
        self.history.clear()

    def report(self, logprob):
        if self.verbose:
            delta = logprob - self.history[-1] if self.history else np.nan
            message = self._template.format(iter=self.iter + 1, logprob=logprob, delta=delta)
            print(message, file=sys.stderr)

        if len(self.history) > 0 and self.history[-1] > logprob:
            msg = 'Log probability is NOT non-decreasing from previous {:.2f} to current {:.2f}'
            raise ValueError(msg.format(self.history[-1], logprob))

        self.history.append(logprob)
        self.iter += 1

    @property
    def converged(self):
        return (self.iter == self.n_iter or 
                (len(self.history) == 2 and
                 self.history[1] - self.history[0] < self.tol))


class HMMConf(utils.Logged):
    def __init__(self, startprob, transcube, emitmat, confmat, distmat, statemap, obsmap, 
                 n_states, n_iter=10, tol=1e-2, verbose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        msg = """Number of {item} are different between state transition cube and emission matrix

        [transcube]: {left}
        [emitmat]: {right}
        """

        obs_msg = msg.format(item='activities', left=transcube.shape[0], right=emitmat.shape[1])
        state_msg = msg.format(item='states', left=transcube.shape[1], right=emitmat.shape[0])
        assert transcube.shape[0] == emitmat.shape[1], obs_msg
        assert transcube.shape[1] == emitmat.shape[0], state_msg

        self.startprob = startprob
        self.transcube = transcube
        self.transcube_d = np.zeros(transcube.shape) + 1. / self.transcube.shape[1]
        self.emitmat = emitmat
        self.emitmat_d = np.zeros(emitmat.shape) + 1. / self.emitmat.shape[1]
        # self.logtranscube = utils.log_mask_zero(self.transcube)
        # self.logtranscube_d = utils.log_mask_zero(self.trans_cube_d)
        # self.logemitmat = utils.log_mask_zero(self.emitmat)
        # self.logemitmat_d = utils.log_mask_zero(self.emitmat_d)
        self.confmat = confmat
        self.distmat = distmat
        self.statemap = statemap
        self.obsmap = obsmap
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.monitor = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)

    def conform(self, stateprob, obs):
        if not stateprob.shape == (1, self.n_states):
            msg = """Number of components are different
            [expected]: {left}
            [parameter]: {right}
            """.format(left=(1, self.n_states), right=stateprob.shape)
            raise ValueError(msg)

        if not np.isclose(stateprob.sum(), [1.]):
            raise ValueError('State estimation: {} does not sum to 1.'.format(stateprob))

        return np.dot(stateprob, self.confmat[obs])

    def emissionprob(self, stateprob, obs):
        """
        Computes P(x is obs at time t | z at time t) where x is the observation variable
        and z is the state variable. 

        :param stateprob: P(Z_t | X_{1:t} = x_{1:t}), i.e. normalized forward probability at time t
        :param obs: observation at time t
        """
        if (stateprob.shape[1] != self.emitmat.shape[0]):
            raise ValueError('Invalid state length: {}'.format(stateprob))
        conf = self.conform(stateprob, obs)

        msg = 'Conformance between state and observation at time t ' \
              'before observation adjustment: {:.2f}'.format(conf[0])
        self.logger.info(msg)

        return conf * self.emitmat[:,obs] + (1 - conf) * self.emitmat_d[:,obs]

    def stateprob(self, stateprob, obs):
        """
        Computes P(z at time t | z at time t - 1, x is obs at time t - 1) where x is the observation
        variable and z is the state variable.

        :param stateprob: P(Z_{t-1} | X_{1:t-1} = x_{1:t-1}), i.e., normalized forward probability at time t  - 1
        :param obs: observed activity at time t - 1
        """
        if (stateprob.shape[1] != self.n_states):
            raise ValueError('Invalid state length: {}'.format(stateprob))

        conf = self.conform(stateprob, obs)

        msg = 'Conformance between state and observation at time t' \
              ' before observation adjustment: {:.2f}'.format(conf[0])
        self.logger.info(msg)

        return conf * self.transcube[obs,:,:] + (1 - conf) * self.transcube_d[obs,:,:]

    def forward(self, obs, prev_obs=None, prev_fwd=None):
        """Computes the log forward probability.
        """
        if prev_fwd is None:
            return utils.log_mask_zero(self.startprob) + utils.log_mask_zero(self.emissionprob(self.startprob, obs))

        work_buffer = prev_fwd.copy()
        utils.log_normalize(work_buffer, axis=1)
        prev_stateprob = np.exp(work_buffer)
        work_buffer = utils.log_mask_zero(self.stateprob(prev_stateprob, prev_obs))
        # print('Previous fwd: {}'.format(prev_fwd))
        cur_fwd_est = logsumexp(work_buffer.T + prev_fwd, axis=1)
        cur_fwd_est = cur_fwd_est.reshape([1, self.n_states])
        work_buffer = cur_fwd_est.copy()
        # print('Current fwd est.: {}'.format(work_buffer))
        utils.log_normalize(work_buffer, axis=1)
        cur_stateprob = np.exp(work_buffer)

        msg0 = '   Log state estimate of time t before observation at time t: {}'.format(cur_fwd_est)
        msg1 = 'W. State estimate of time t before observation at time t: {}'.format(cur_stateprob)
        self.logger.info(msg0)
        self.logger.info(msg1)

        obs_prob = self.emissionprob(cur_stateprob, obs)
        msg2 = 'Likelihood of observation at states time t: {}'.format(obs_prob)
        obs_prob = utils.log_mask_zero(obs_prob)
        self.logger.info(msg2)
        return obs_prob + cur_fwd_est

    def _do_forward_pass(self, X, prev_fwd=None):
        pass

    def _do_backward_pass(self, X, conf, prev_bwd=None):
        pass

    def fit(self, X, lengths):
        pass

    def _compute_posteriors(fwdlattice, bwdlattice):
        pass
