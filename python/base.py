from collections import deque
from sklearn.base import _pprint
from scipy.misc import logsumexp
import numpy as np
import pandas as pd
import utils, warnings, sys, time, string


np.set_printoptions(precision=16)


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
        self.logger = utils.make_logger(self.__class__.__name__)

    def __repr__(self):
        class_name = self.__class__.__name__
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


class HMMConf:
    """Modified HMM that computes the conformance of a stream of event data and updates state
    estimation considering the previous state and conformance. The dynamic bayesian network is 
    modified so that the next state depends on both the last state and last observation. 

    **Parameters**:
    :param startprob array_like: vector of initial state distribution
    :param transcube array_like: cube denoting state-transition probability extracted from the reachability graph of net model
    :param emitmat array_like: emission probability matrix extracted from the reachability graph of net model
    :param distmat array_like: state distance matrix extracted from the reachability graph of net model
    :param int2state dict: mapping from state index to state string representation
    :param int2obs dict: mapping from observation index to observation string representation
    :param n_states int: number of states in HMM
    :param n_obs int: number of observation in HMM
    :param params string, optional: Controls which parameters are updated during the training phase. Can contain any combination of 's' for startprob, 't' for transcube, and 'o' for emitmat. Defaults to 'to'.
    :param n_iter int, optional: Maximum number of iterations to perform in EM parameter estimation
    :param tol float, optional: The convergence threshold. EM will stop once the gain between iterations is below this value.
    :param verbose bool: When ``True`` per-iteration convergence reports are printed to :data:`sys.stderr`. 
    """
    FIRST_IND = 0
    SECOND_IND = 1
    UPDATED_IND = 2

    def __init__(self, startprob, transcube, emitmat, confmat, distmat, 
                 int2state, int2obs, n_states, n_obs, params='to',
                 n_iter=10, tol=1e-2, verbose=False, *args, **kwargs): 
        utils.assert_shape('activities', transcube.shape[0], emitmat.shape[1])
        utils.assert_shape('states', transcube.shape[1], emitmat.shape[0])
        utils.assert_no_negatives('transcube', transcube)
        utils.assert_no_negatives('emitmat', emitmat)

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
        """Computes the conformance of an observation with respect to a state estimation.

        :param stateprob array_like: state estimation vector that sums to 1.
        :param obs int: observation
        """
        utils.assert_shape('stateprob', (1, self.n_states), stateprob.shape)

        if not np.isclose(stateprob.sum(), [1.]):
            raise ValueError('State estimation: {} does not sum to 1.'.format(stateprob))

        v = np.dot(stateprob, self.confmat[obs])
        utils.assert_no_negatives('conformance', v)
        utils.assert_bounded('conformance', v, 0, 1)
        return v

    def emissionprob(self, obs, conf):
        """
        Computes P(x is obs at time t | z at time t) where x is the observation variable
        and z is the state variable. 

        :param obs int: observation at time t
        :param conf float: conformance between stateprob and obs
        """
        self.logger.debug('conform: {}'.format(conf))
        self.logger.debug('emitmat_d: {}'.format(self.emitmat_d[:,obs]))
        prob = conf * self.emitmat[:,obs] + (1 - conf) * self.emitmat_d[:,obs]
        return prob

    def stateprob(self, obs, conf):
        """
        Computes P(z at time t | z at time t - 1, x is obs at time t - 1) where x is the observation
        variable and z is the state variable.

        :param obs int: observed activity at time t - 1
        :param conf float: conformance between stateprob and obs
        """
        prob = conf * self.transcube[obs,:,:] + (1 - conf) * self.transcube_d[obs,:,:]
        utils.assert_no_negatives('transcube[{},:,:]'.format(obs), self.transcube[obs,:,:])
        utils.assert_no_negatives('transcube_d[{},:,:]'.format(obs), self.transcube_d[obs,:,:])
        utils.assert_no_negatives('stateprob, conform: {}'.format(conf), prob)
        return prob

    def _forward(self, obs, prev_obs=None, prev_fwd=None):
        """Computes the log forward probability.

        :param obs int: observation
        :param prev_obs int, optional: previous observation if any
        :param prev_fwd array_like, optional: previous log forward probability for all states
        :return: log forward probability, conformance array, log state probability, log emission probability
        """
        conf_arr = np.full(3, -1.)

        if prev_fwd is None:
            self.logger.debug('startprob: {}'.format(self.startprob))
            emitconf = self.conform(self.startprob, obs)
            obsprob = self.emissionprob(obs, emitconf)
            logobsprob = utils.log_mask_zero(obsprob)
            logfwd = utils.log_mask_zero(self.startprob) + logobsprob
            fwd = logfwd.copy()
            utils.log_normalize(fwd, axis=1) # to avoid underflow
            fwd = np.exp(fwd)
            utils.normalize(fwd, axis=1) # to avoid not summing to 1 after exp
            conf_arr[self.SECOND_IND] = emitconf[0]
            conf_arr[self.UPDATED_IND] = self.conform(fwd, obs)
            logstateprob = np.full((self.transcube.shape[1], self.transcube.shape[1]), -np.inf)
            return logfwd, conf_arr, logstateprob, logobsprob

        arr_buffer = prev_fwd.copy()
        utils.log_normalize(arr_buffer, axis=1) # to avoid underflow
        # P(Z_{t-1} | X_{1:t-1} = x_{1:t-1}), i.e., normalized forward probability at time t  - 1
        prev_stateprob = np.exp(arr_buffer)
        utils.normalize(prev_stateprob, axis=1) # to avoid not summing to 1 after exp
        stateconf = self.conform(prev_stateprob, prev_obs)
        stateprob = self.stateprob(prev_obs, stateconf)
        logstateprob = utils.log_mask_zero(stateprob)
        n_nonzeros = np.count_nonzero(stateprob)
        self.logger.debug('No. non-zero vals: {}'.format(n_nonzeros))
        self.logger.debug('Previous fwd: \n{}'.format(prev_fwd))
        self.logger.debug('state prob: \n{}'.format(stateprob))
        self.logger.debug('log state prob: \n{}'.format(logstateprob))
        work_buffer = logstateprob.T + prev_fwd
        self.logger.debug('No. nan vals: {}'.format(np.count_nonzero(np.isnan(work_buffer))))
        cur_fwd_est = logsumexp(work_buffer, axis=1)
        cur_fwd_est = cur_fwd_est.reshape([1, self.n_states])
        # print('Current fwd est.: {}'.format(work_buffer))

        # some helpful loggings during development...
        arr_buffer = cur_fwd_est.copy()
        utils.log_normalize(arr_buffer, axis=1) # to avoid underflow
        # P(Z_t | X_{1:t} = x_{1:t}), i.e. normalized forward probability at time t
        cur_stateprob = np.exp(arr_buffer)
        utils.normalize(cur_stateprob, axis=1) # to avoid not summing to 1 after exp

        msg0 = '   Log state estimate of time t before observation at time t: \n{}'.format(cur_fwd_est)
        msg1 = 'W. State estimate of time t before observation at time t: \n{}'.format(cur_stateprob)
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
        utils.log_normalize(fwd, axis=1) # to avoid underflow
        fwd = np.exp(fwd)
        utils.normalize(fwd, axis=1) # to avoid not summing to 1 after exp

        conf_arr[self.FIRST_IND] = stateconf[0]
        conf_arr[self.SECOND_IND] = emitconf[0]
        conf_arr[self.UPDATED_IND] = self.conform(fwd, obs)

        return logfwd, conf_arr, logstateprob, logobsprob

    def forward(self, obs, prev_obs=None, prev_fwd=None):
        """Computes the log forward probability.

        :param obs int: observation
        :param prev_obs int, optional: previous observation if any
        :param prev_fwd array_like, optional: previous log forward probability for all states
        :return: log forward probability, conformance array
        """
        logfwd, conf_arr, _, _ = self._forward(obs, prev_obs, prev_fwd)
        return logfwd, conf_arr

    def backward(self, obs, prev_obs, conf_arr, prev_bwd=None):
        """Computes the log backward probability.

        :param obs int: observation
        :param prev_obs int: previous observation
        :param conf_arr array_like: conformance vector computed for prev_obs using forward probability
        :param prev_bwd array_like, optional: previous log backward probability 
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
        :return: log likelihood, the forward lattice, the conformance lattice, state-transition and observation lattice
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
        """Computes the forward lattice containing the forward probability of a single sequence of
        observations.

        :param X: array of observations
        :type X: array_like (n_samples, 1)
        :return: log likelihood, the forward lattice, the conformance lattice
        """
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
        """Estimate model parameters using EM.

        :param X array_like, shape (n_samples, 1): sample data
        :param lengths array_like, shape (n_sequences,): lengths of the individual sequences in ``X``. The sum of these should be ``n_samples``.
        :return: ``self``
        """
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

        :param stats dict: dictionary storing the sufficient statistics of the HMM 
        :param logstateprob array_like: Log state probability at each time frame 1 to T
        :param logobsprob array_like: Log observation probability at each time frame 1 to T
        :param conflattice array_like: Conformance at each time frame 1 to T
        :param posteriors array_like: Posterior likelihood at each time frame 1 to T
        :param fwdlattice array_like: Log forward probability at each time frame 1 to T
        :param bwdlattice array_like: Log backward probability at each time frame 1 to T
        """
        stats['nobs'] += 1

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
        """Initialize sufficient statistics.

        :return: sufficient statistics
        """
        stats = {'nobs': 0,
                 'start': np.zeros(self.n_states),
                 'trans': np.zeros(self.transcube.shape),
                 'obs': np.zeros(self.emitmat.shape)}
        return stats

    def _do_mstep(self, stats):
        """M step of EM 

        :param stats dict: sufficient statistics
        """
        if 's' in self.params:
            startprob = self.startprob + stats['start']
            self.startprob = np.where(self.startprob == 0.,
                                      self.startprob, startprob)
            utils.normalize(self.startprob, axis=1)

        if 't' in self.params:
            # Note:
            # It is technically incorrect to add the previous transcube_d to the newly estimated parameter
            # values. However, this is practically necessary since we create parameter estimation updates
            # for all value even if they did not appear in the sample data used for the EM update since 
            # we always loop over all parameter values for ease of implementation. 
            # For example, if transcube_d[a,i,j] does not appear in the EM data, our learnt parameter estimation would 
            # yield transcube_d[a,i,j] = 0 by default. With little sample data, this will mean that transcube_d will 
            # not be a probability matrix. This in turn will mess up state estimation if conformance = 0, 
            # because it will yield a zero vector. 
            # To avoid the above scenario, from get go we add the initially uniform probability matrix so
            # that the resulting transcube_d will remain a probability matrix if a row has all 0s in the parameter
            # estimation. 
            # In the contrary case where some value in row i transcube_d[a,i,j] has a non-zero parameter estimation,
            # we will still get the same effect since we will normalize the row later anyway.
            self.transcube_d = stats['trans'] + self.transcube_d
            utils.normalize(self.transcube_d, axis=2)   # normalize row
            utils.assert_no_negatives('transcube_d', self.transcube_d)

        if 'o' in self.params:
            # See the above explanation
            self.emitmat_d = stats['obs'] + self.emitmat_d
            utils.normalize(self.emitmat_d, axis=1)
            utils.assert_no_negatives('emitmat_d', self.emitmat_d)

    def _compute_posteriors(self, fwdlattice, bwdlattice):
        """Posterior likelihood of states given data.

        :param fwdlattice array_like: log forward probability 
        :param bwdlattice array_like: log backward probability
        """
        log_gamma = fwdlattice + bwdlattice
        utils.log_normalize(log_gamma, axis=1)  # this prevents underflow
        with np.errstate(under='ignore'):
            return np.exp(log_gamma)

    def compute_expected_distance(self, obs, logfwd, 
                                  prev_obs=None, prev_logfwd=None):
        """Compute expected distance between current state estimation derived from log forward probability and initial state estimation.

        :param obs int: observation
        :param logfwd array_like: log forward probability
        :param prev_obs int: previous observation
        :param prev_logfwd array_like: previous log forward probability
        :return: expected distance
        :rtype: float
        """
        if prev_obs is None:
            emitconf = self.conform(self.startprob, obs)
            obsprob = self.emissionprob(obs, emitconf)
            logobsprob = utils.log_mask_zero(obsprob)
            work_buffer = logfwd.copy()
            utils.log_normalize(work_buffer, axis=1)
            work_buffer = np.exp(work_buffer)[:,np.newaxis]
            work_buffer = self.distmat * work_buffer
            return work_buffer.sum(axis=None)

        arr_buffer = prev_logfwd.copy()
        utils.log_normalize(arr_buffer, axis=1) # to avoid underflow
        # P(Z_{t-1} | X_{1:t-1} = x_{1:t-1}), i.e., normalized forward probability at time t  - 1
        prev_stateprob = np.exp(arr_buffer)
        utils.normalize(prev_stateprob, axis=1) # to avoid not summing to 1 after exp

        # w_{i,j}(x_t)
        stateconf = self.conform(prev_stateprob, prev_obs)
        stateprob = self.stateprob(prev_obs, stateconf)
        logstateprob = utils.log_mask_zero(stateprob)

        # v_{j}(x_{t+1})
        work_buffer = logstateprob.T + prev_logfwd
        cur_fwd_est = logsumexp(work_buffer, axis=1)
        cur_fwd_est = cur_fwd_est.reshape([1, self.n_states])
        arr_buffer = cur_fwd_est.copy()
        utils.log_normalize(arr_buffer, axis=1) 
        # P(Z_t | X_{1:t} = x_{1:t}), i.e. normalized forward probability at time t
        cur_stateprob = np.exp(arr_buffer)
        utils.normalize(cur_stateprob, axis=1) 
        emitconf = self.conform(cur_stateprob, obs)
        obsprob = self.emissionprob(obs, emitconf)
        logobsprob = utils.log_mask_zero(obsprob)[np.newaxis,:]
        prev_logfwd = prev_logfwd.ravel()[:,np.newaxis]

        utils.assert_shape('logobsprob', (1, self.n_states), logobsprob.shape)
        utils.assert_shape('logstateprob', (self.n_states, self.n_states), logstateprob.shape)
        utils.assert_shape('prev_logfwd', (self.n_states, 1), prev_logfwd.shape)

        work_buffer = logobsprob + logstateprob + prev_logfwd - logsumexp(logfwd, axis=1)
        work_buffer = np.exp(work_buffer)
        utils.assert_shape('node pmf', self.distmat.shape, work_buffer.shape)
        work_buffer = np.multiply(self.distmat, work_buffer)
        return work_buffer.sum(axis=None)
