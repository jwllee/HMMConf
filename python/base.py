from collections import deque
from sklearn.base import _pprint
from scipy.misc import logsumexp
import numpy as np
import pandas as pd
import utils, warnings, sys, time, string


class ConvergenceMonitor:
    _header = '{:>10s} {:>16s} {:>16s}'.format('iteration', 'logprob', 'delta')
    _template = "{iter:>10d} {logprob:>16.4f} {delta:>+16.4f}"

    def __init__(self, tol, n_iter, verbose):
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.history = deque(maxlen=2)
        self.iter = 0
        self.logger = utils.make_logger(self.__class__.__name__)

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
            if delta == np.nan:
                print(self._header)
            message = self._template.format(iter=self.iter + 1, logprob=logprob, delta=delta)
            print(message, file=sys.stderr)

        if len(self.history) > 0 and self.history[-1] > logprob:
            msg = 'Log probability is NOT non-decreasing from previous {:.2f} to current {:.2f}'
            raise ValueError(msg.format(self.history[-1], logprob))

        self.history.append(logprob)
        self.iter += 1

    @property
    def converged(self):
        converged_ = (self.iter == self.n_iter or 
                     (len(self.history) == 2 and
                     self.history[1] - self.history[0] < self.tol))
        return converged_


class HMMConf:
    FIRST_IND = 0
    SECOND_IND = 1
    UPDATED_IND = 2

    def __init__(self, startprob, transcube, emitmat, confmat, distmat, 
                 int2state, int2obs, n_states, n_obs, params='to',
                 n_iter=10, tol=1e-2, verbose=False, *args, **kwargs): 
        utils.assert_shape('activities', transcube.shape[0], emitmat.shape[1])
        utils.assert_shape('states', transcube.shape[1], emitmat.shape[0])

        self.logger = utils.make_logger(self.__class__.__name__)
        self.params = params
        self.startprob = startprob
        self.transcube = transcube
        self.transcube_d = np.zeros(transcube.shape) + 1. / self.transcube.shape[1]
        self.emitmat = emitmat
        self.emitmat_d = np.zeros(emitmat.shape) + 1. / self.emitmat.shape[1]
        self.confmat = confmat
        self.distmat = distmat
        self.int2state = int2state
        self.int2obs = int2obs
        self.n_states = n_states
        self.n_obs = n_obs
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
        self.logger.debug('conform: {}'.format(conf))
        self.logger.debug('emitmat_d: {}'.format(self.emitmat_d[:,obs]))
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

    def _forward(self, obs, prev_obs=None, prev_fwd=None):
        """Computes the log forward probability.

        :return: log forward probability, conformance array
        """
        conf_arr = np.full(3, -1.)

        if prev_fwd is None:
            self.logger.debug('startprob: {}'.format(self.startprob))
            emitconf = self.conform(self.startprob, obs)
            obsprob = self.emissionprob(obs, emitconf)
            logobsprob = utils.log_mask_zero(obsprob)
            logfwd = utils.log_mask_zero(self.startprob) + logobsprob
            fwd = logfwd.copy()
            utils.log_normalize(fwd, axis=1)
            fwd = np.exp(fwd)
            conf_arr[self.SECOND_IND] = emitconf[0]
            conf_arr[self.UPDATED_IND] = self.conform(fwd, obs)
            logstateprob = np.full((self.transcube.shape[1], self.transcube.shape[1]), -np.inf)
            return logfwd, conf_arr, logstateprob, logobsprob

        arr_buffer = prev_fwd.copy()
        utils.log_normalize(arr_buffer, axis=1)
        # P(Z_{t-1} | X_{1:t-1} = x_{1:t-1}), i.e., normalized forward probability at time t  - 1
        prev_stateprob = np.exp(arr_buffer)
        stateconf = self.conform(prev_stateprob, prev_obs)
        stateprob = self.stateprob(prev_obs, stateconf)
        logstateprob = utils.log_mask_zero(stateprob)
        # print('Previous fwd: {}'.format(prev_fwd))
        self.logger.debug('State trans probability: \n{}'.format(stateprob))
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
        self.logger.debug(msg0)
        self.logger.debug(msg1)

        emitconf = self.conform(cur_stateprob, obs)
        obsprob = self.emissionprob(obs, emitconf)
        logobsprob = utils.log_mask_zero(obsprob)

        msg2 = 'Likelihood of observation at states time t: {}'.format(obsprob)
        self.logger.debug(msg2)
        msg3 = 'Conformance between state and observation at time t ' \
              'before observation adjustment: {:.2f}'.format(emitconf[0])
        self.logger.debug(msg3)

        logfwd = logobsprob + cur_fwd_est
        fwd = logfwd.copy()
        utils.log_normalize(fwd, axis=1)
        fwd = np.exp(fwd)

        conf_arr[self.FIRST_IND] = stateconf[0]
        conf_arr[self.SECOND_IND] = emitconf[0]
        conf_arr[self.UPDATED_IND] = self.conform(fwd, obs)

        return logfwd, conf_arr, logstateprob, logobsprob

    def forward(self, obs, prev_obs=None, prev_fwd=None):
        logfwd, conf_arr, _, _ = self._forward(obs, prev_obs, prev_fwd)
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
            logbwd = logsumexp(logemitprob + logstateprob, axis=1)
        else:
            logbwd = logsumexp(logemitprob + logstateprob + prev_bwd, axis=1)

        self.logger.debug('Backward probability: \n{}'.format(logbwd))

        return logbwd

    def _do_forward_pass(self, X):
        """Computes the forward lattice containing the forward probability of a single sequence of
        observations.

        :param X: array of observations
        :type X: array_like (n_samples, 1)
        :return: the forward lattice, the conformance lattice, state-transition and observation lattice
        """
        n_samples = X.shape[0]
        fwdlattice = np.ndarray((n_samples, self.n_states))
        conflattice = np.ndarray((n_samples, 3))
        framelogstateprob = np.ndarray((n_samples, self.n_states, self.n_states))
        framelogobsprob = np.ndarray((n_samples, self.n_states))

        times = []

        # first observation
        obs = X[0,0]
        start = time.time()
        fwd, conf, _, logobsprob = self._forward(obs)
        end = time.time()
        times.append(end - start)
        fwdlattice[0] = fwd
        conflattice[0] = conf
        framelogstateprob[0] = -1.
        framelogobsprob[0] = logobsprob

        prev_obs = obs
        prev_fwd = fwd
        for i in range(1, n_samples):
            obs = X[i,0]
            start = time.time()
            fwd, conf, logstateprob, logobsprob = self._forward(obs, prev_obs, prev_fwd)
            end = time.time()
            times.append(end - start)
            fwdlattice[i] = fwd
            conflattice[i] = conf
            framelogstateprob[i] = logstateprob
            framelogobsprob[i] = logobsprob

            prev_obs = obs
            prev_fwd = fwd

        self.logger.debug('Average forward probability time: {:.4f}s'.format(np.mean(times)))

        with np.errstate(under='ignore'):
            logprob = logsumexp(fwdlattice[-1])
            return logprob, fwdlattice, conflattice, framelogstateprob, framelogobsprob

    def do_forward_pass(self, X):
        logprob, fwdlattice, conflattice, logstateprob, logobsprob = self._do_forward_pass(X)
        return logprob, fwdlattice, conflattice

    def do_backward_pass(self, X, conflattice):
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
        self.monitor._reset()

        for it in range(self.n_iter):
            cur_logprob = 0
            stats = self._initialize_sufficient_statistics()

            for i, j in utils.iter_from_X_lengths(X, lengths):
                logprob, fwdlattice, conflattice, logstateprob, logobsprob = self._do_forward_pass(X[i:j])
                cur_logprob += logprob
                bwdlattice = self.do_backward_pass(X[i:j], conflattice)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)

                self._accumulate_sufficient_statistics(stats, X[i:j], logstateprob, logobsprob, conflattice,
                                                       posteriors, fwdlattice, bwdlattice)

            self._do_mstep(stats)

            self.monitor.report(cur_logprob)
            if self.monitor.converged:
                msg = 'Converged at iteration {} with current logprob {:.2f} and previous logprob {:.2f}'
                cur_logprob = self.monitor.history[1] if len(self.monitor.history) > 1 else self.monitor.history[0]
                prev_logprob = self.monitor.history[1] if len(self.monitor.history) > 1 else -1
                msg = msg.format(it, cur_logprob, prev_logprob)
                self.logger.debug(msg)
                break

        return self

    def _accumulate_sufficient_statistics(self, stats, X, logstateprob, logobsprob, conflattice, 
                                          posteriors, fwdlattice, bwdlattice):
        """Updates sufficient statistics from a given sample.

        """
        if 's' in self.params:
            stats['start'] += posteriors[0]

        if 't' in self.params:
            n_samples = logobsprob.shape[0]

            if n_samples <= 1:
                return

            n_states, n_obs = self.n_states, self.n_obs

            utils.assert_shape('logstateprob', (n_samples, n_states, n_states), logstateprob.shape)
            utils.assert_shape('logobsprob', (n_samples, n_states), logobsprob.shape)

            log_xi_sum = np.full((self.n_obs, self.n_states, self.n_states), -np.inf)

            for t in range(n_samples - 1):
                # skip events that are perfectly conforming
                if conflattice[t, self.UPDATED_IND] >= 1.:  
                    continue

                o = X[t]    # to identify the state transition matrix to update
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        to_add = (fwdlattice[t, i] + logstateprob[t, i, j]
                                  + logobsprob[t + 1, j] + bwdlattice[t + 1, j])
                        log_xi_sum[o, i, j] = np.logaddexp(log_xi_sum[o, i, j], to_add)

            denominator = logsumexp(log_xi_sum, axis=2)

            utils.assert_shape('denominator', (n_obs, n_states), denominator.shape)

            for o in range(self.n_obs):
                # only update values if sample affected the state-transition between i and j for o
                to_update = denominator[o,:] != -np.inf
                # log_xi_sum[o,:] = np.subtract(log_xi_sum[o,:], denominator[o,:], where=to_update)
                np.subtract(log_xi_sum[o,:], denominator[o,:], out=log_xi_sum[o,:], where=to_update)

            with np.errstate(under='ignore'):
                stats['trans'] += np.exp(log_xi_sum)

        if 'o' in self.params:
            n_samples = logobsprob.shape[0]

            i = 0
            for t, symbol in enumerate(np.concatenate(X)):
                # skip if it's perfectly conforming
                if conflattice[t, self.UPDATED_IND] >= 1.:
                    continue

                self.logger.debug('No. of non-zeros: {}'.format(np.count_nonzero(posteriors[t])))
                stats['obs'][:, symbol] += posteriors[t]
                i += 1

            self.logger.debug('There were {} non-conforming events'.format(i))

            denominator = stats['obs'].sum(axis=1)[:, np.newaxis]
            # avoid zero division by replacement as 1.
            # denominator[denominator == 0.] = 1.
            # stats['obs'] /= denominator

            # only update matrix if the sample affected the emission probability
            to_update = denominator != 0.
            np.divide(stats['obs'], denominator, out=stats['obs'], where=to_update)

    def _initialize_sufficient_statistics(self):
        stats = {'nobs': 0,
                 'start': np.zeros(self.n_states),
                 'trans': np.zeros(self.transcube.shape),
                 'obs': np.zeros(self.emitmat.shape)}
        return stats

    def _do_mstep(self, stats):
        if 's' in self.params:
            startprob = self.startprob + stats['start']
            self.startprob = np.where(self.startprob == 0.,
                                      self.startprob, startprob)
            utils.normalize(self.startprob, axis=1)

        if 't' in self.params:
            # self.transcube_d = np.where(self.transcube_d == 0.,
            #                             self.transcube_d, transcube)
            self.transcube_d = stats['trans'] + self.transcube_d
            utils.normalize(self.transcube_d, axis=2)

        if 'o' in self.params:
            self.emitmat_d = stats['obs'] + self.emitmat_d
            utils.normalize(self.emitmat_d, axis=1)

    def _compute_posteriors(self, fwdlattice, bwdlattice):
        log_gamma = fwdlattice + bwdlattice
        utils.log_normalize(log_gamma, axis=1)  # this prevents underflow
        with np.errstate(under='ignore'):
            return np.exp(log_gamma)
