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

    def emissionprob(self, obs, conf):
        """
        Computes P(x is obs at time t | z at time t) where x is the observation variable
        and z is the state variable. 

        :param obs: observation at time t
        :param conf: conformance between stateprob and obs
        """
        prob = conf * self.emitmat[:,obs] + (1 - conf) * self.emitmat_d[:,obs]
        return prob

    def stateprob(self, obs, conf):
        """
        Computes P(z at time t | z at time t - 1, x is obs at time t - 1) where x is the observation
        variable and z is the state variable.

        :param obs: observed activity at time t - 1
        :param conf: conformance between stateprob and obs
        """
        prob = conf * self.transcube[obs,:,:] + (1 - conf) * self.transcube_d[obs,:,:]
        return prob

    def forward(self, obs, prev_obs=None, prev_fwd=None):
        """Computes the log forward probability.

        :return: log forward probability, conformance array
        """
        conf_arr = np.full(3, -1.)

        if prev_fwd is None:
            emitconf = self.conform(self.startprob, obs)
            emitprob = self.emissionprob(obs, emitconf)
            logfwd = utils.log_mask_zero(self.startprob) + utils.log_mask_zero(emitprob)
            fwd = logfwd.copy()
            utils.log_normalize(fwd, axis=1)
            fwd = np.exp(fwd)
            conf_arr[1] = emitconf[0]
            conf_arr[2] = self.conform(fwd, obs)
            return logfwd, conf_arr

        arr_buffer = prev_fwd.copy()
        utils.log_normalize(arr_buffer, axis=1)
        # P(Z_{t-1} | X_{1:t-1} = x_{1:t-1}), i.e., normalized forward probability at time t  - 1
        prev_stateprob = np.exp(arr_buffer)
        stateconf = self.conform(prev_stateprob, prev_obs)
        stateprob = self.stateprob(prev_obs, stateconf)
        logstateprob = utils.log_mask_zero(stateprob)
        # print('Previous fwd: {}'.format(prev_fwd))
        self.logger.info('State trans probability: \n{}'.format(stateprob))
        cur_fwd_est = logsumexp(logstateprob.T + prev_fwd, axis=1)
        cur_fwd_est = cur_fwd_est.reshape([1, self.n_states])
        # print('Current fwd est.: {}'.format(work_buffer))

        # some helpful loggings during development...
        arr_buffer = cur_fwd_est.copy()
        utils.log_normalize(arr_buffer, axis=1)
        # P(Z_t | X_{1:t} = x_{1:t}), i.e. normalized forward probability at time t
        cur_stateprob = np.exp(arr_buffer)

        msg0 = '   Log state estimate of time t before observation at time t: {}'.format(cur_fwd_est)
        msg1 = 'W. State estimate of time t before observation at time t: {}'.format(cur_stateprob)
        self.logger.info(msg0)
        self.logger.info(msg1)

        emitconf = self.conform(cur_stateprob, obs)
        obsprob = self.emissionprob(obs, emitconf)
        msg2 = 'Likelihood of observation at states time t: {}'.format(obsprob)
        obsprob = utils.log_mask_zero(obsprob)
        self.logger.info(msg2)
        msg3 = 'Conformance between state and observation at time t ' \
              'before observation adjustment: {:.2f}'.format(emitconf[0])
        self.logger.info(msg3)

        logfwd = obsprob + cur_fwd_est
        fwd = logfwd.copy()
        utils.log_normalize(fwd, axis=1)
        fwd = np.exp(fwd)

        conf_arr[0] = stateconf[0]
        conf_arr[1] = emitconf[0]
        conf_arr[2] = self.conform(fwd, obs)

        return logfwd, conf_arr

    def backward(self, obs, prev_obs, conf_arr, prev_bwd=None):
        """Computes the log backward probability.

        :return: log backward probability
        """
        emitprob = self.emissionprob(obs, conf_arr[1])
        stateprob = self.stateprob(prev_obs, conf_arr[0])
        logemitprob = utils.log_mask_zero(emitprob)
        logstateprob = utils.log_mask_zero(stateprob)

        # no need to transpose logstateprob since we are broadcasting addition across the rows
        summed = logemitprob + logstateprob
        if prev_bwd is None:
            bwd = logsumexp(logemitprob + logstateprob, axis=1)
        else:
            bwd = logsumexp(logemitprob + logstateprob + prev_bwd, axis=1)

        self.logger.info('Backward probability: \n{}'.format(bwd))

        return bwd

    def _do_forward_pass(self, X):
        """Computes the forward lattice containing the forward probability of a single sequence of
        observations.

        :param X: array of observations
        :type X: array_like (n_samples, 1)
        :return: the forward lattice and the conformance lattice
        """
        n_samples = X.shape[0]
        fwdlattice = np.ndarray((n_samples, self.n_states))
        conflattice = np.ndarray((n_samples, 3))

        # first observation
        obs = X[0,0]
        fwd, conf = self.forward(obs)
        fwdlattice[0] = fwd
        conflattice[0] = conf

        prev_obs = obs
        prev_fwd = fwd
        for i in range(1, n_samples):
            obs = X[i,0]
            fwd, conf = self.forward(obs, prev_obs, prev_fwd)
            fwdlattice[i] = fwd
            conflattice[i] = conf

            prev_obs = obs
            prev_fwd = fwd

        return fwdlattice, conflattice

    def _do_backward_pass(self, X, conflattice):
        """Computes the backward lattice containing the backward probability of a single sequence 
        of observations.

        :param X: array of observations
        :type X: array_like (n_samples, 1)
        :param conflattice: 
        :return: the backward lattice
        """
        n_samples = X.shape[0]
        bwdlattice = np.ndarray((n_samples, self.n_states))
        
        # last observation bwd(T) = 1. for all states
        bwdlattice[-1,:] = 0.
        
        obs = X[-1,0]
        prev_bwd = bwdlattice[-1]
        for i in range(n_samples - 2, -1, -1): # compute bwd(T - 1) to bwd(1)
            prev_obs = X[i,0]
            conf_arr = conflattice[i+1]
            bwd = self.backward(obs, prev_obs, conf_arr, prev_bwd)
            bwdlattice[i] = bwd

            obs = prev_obs
            prev_bwd = bwd

        return bwdlattice

    def fit(self, X, lengths):
        pass

    def _compute_posteriors(self, fwdlattice, bwdlattice):
        log_gamma = fwdlattice + bwdlattice
        utils.log_normalize(log_gamma, axis=1)
        with np.errstate(under='ignore'):
            return np.exp(log_gamma), np.exp(log_gamma).sum(axis=1)
