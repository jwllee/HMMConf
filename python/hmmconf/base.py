from collections import deque
from sklearn.base import _pprint
from scipy.special import logsumexp
import numpy as np
import pandas as pd
import warnings, sys, time, string
from . import utils
import multiprocessing as mp


__all__ = [
    'ConvergenceMonitor',
    'HMMConf'
]


np.set_printoptions(precision=16)


logger = utils.make_logger(__file__)


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
    :param conform_f function: conformance function
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
    :param n_jobs int: the number of jobs to run in parallel for EM fitting. None means 1 and -1 means use all processors.
    """
    FIRST_IND = 0
    SECOND_IND = 1
    UPDATED_IND = 2

    def __init__(self, conform_f, startprob, transcube, emitmat, confmat, distmat, 
                 int2state, int2obs, n_states, n_obs, params='to', n_iter=10, 
                 tol=1e-2, verbose=False, n_jobs=None, random_seed=123, *args, **kwargs): 
        utils.assert_shape('activities', transcube.shape[0], emitmat.shape[1])
        utils.assert_shape('states', transcube.shape[1], emitmat.shape[0])
        utils.assert_no_negatives('transcube', transcube)
        utils.assert_no_negatives('emitmat', emitmat)

        self.logger = utils.make_logger(self.__class__.__name__)
        self.conform_f = conform_f
        self.params = params
        self.random_seed = random_seed
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
        self.n_jobs = n_jobs

    def forward(self, obs, prev_obs=None, prev_fwd=None):
        """Computes the log forward probability.

        :param obs int: observation
        :param prev_obs int, optional: previous observation if any
        :param prev_fwd array_like, optional: previous log forward probability for all states
        :return: log forward probability, conformance array
        """
        # logfwd, conf_arr, _, _ = self._forward(obs, prev_obs, prev_fwd)
        logfwd, conf_arr, _, _ = _forward(self.n_states, self.transcube, self.transcube_d, 
                                          self.emitmat, self.emitmat_d, self.confmat, 
                                          self.conform_f, obs, prev_obs, prev_fwd, self.startprob)
        return logfwd, conf_arr

    def _do_forward_pass(self, x):
        """computes the forward lattice containing the forward probability of a single sequence of
        observations.

        :param x: array of observations
        :type x: array_like (n_samples, 1)
        :return: log likelihood, the forward lattice, the conformance lattice, state-transition and observation lattice
        """
        conform_f = self.conform_f
        _do_forward_pass(x, self.n_states, self.transcube, self.transcube_d, self.emitmat,
                         self.emitmat_d, self.confmat, self.startprob, conform_f)

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
        return do_backward_pass(X, conflattice, self.n_states, self.emitmat, self.emitmat_d, 
                                self.transcube, self.transcube_d)

    def __partition_X(self, X, lengths, n):
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

    def __fit_mp(self, X, lengths):
        # parallelize per case
        n_proc = self.n_jobs if self.n_jobs is not None else 1
        X_parts, lengths_parts = self.__partition_X(X, lengths, n_proc)
        n_proc = len(X_parts)

        args_list = [{
            'conform_f': self.conform_f,
            'params': self.params,
            'n_obs': self.n_obs,
            'n_states': self.n_states,
            'transcube': self.transcube,
            'transcube_d': self.transcube_d,
            'emitmat': self.emitmat,
            'emitmat_d': self.emitmat_d,
            'confmat': self.confmat,
            'startprob': self.startprob,
            'X': X_parts[i],
            'lengths': lengths_parts[i]
        } for i in range(n_proc)]

        if n_proc == 1:
            # no need multiprocessing
            logprob, stats = _fit_worker(args_list[0])
            results = [(logprob, stats)]
        else:
            # self.logger.debug('Parallel fit using {} processes'.format(n_proc))
            pool = mp.Pool(processes=n_proc)
            results = pool.map(_fit_worker, args_list)
            pool.close()

        return results

    def __fit(self, X, lengths):
        args = {
            'conform_f': self.conform_f,
            'params': self.params,
            'n_obs': self.n_obs,
            'n_states': self.n_states,
            'transcube': self.transcube,
            'transcube_d': self.transcube_d,
            'emitmat': self.emitmat,
            'emitmat_d': self.emitmat_d,
            'confmat': self.confmat,
            'startprob': self.startprob,
            'X': X,
            'lengths': lengths
        }
        logprob, stats = _fit_worker(args)
        results = [(logprob, stats)]
        return results

    def fit(self, X, lengths):
        """Estimate model parameters using EM.

        :param X array_like, shape (n_samples, 1): sample data
        :param lengths array_like, shape (n_sequences,): lengths of the individual sequences in ``X``. The sum of these should be ``n_samples``.
        :return: ``self``
        """
        self.monitor._reset()

        for it in range(self.n_iter):
            cur_logprob = 0
            stats = _initialize_sufficient_statistics(self.n_states, 
                                                      self.transcube.shape,
                                                      self.emitmat.shape)

            if self.n_jobs is None or self.n_jobs == 1:
                results = self.__fit(X, lengths)
            else:
                results = self.__fit_mp(X, lengths)

            for i in range(len(results)):
                logprob_i, stats_i = results[i]
                stats['trans'] += stats_i['trans']
                stats['obs'] += stats_i['obs']
                stats['start'] += stats_i['start']
                stats['nobs'] += stats_i['nobs']
                cur_logprob += logprob_i

            self._do_mstep(stats)

            self.monitor.report(cur_logprob)
            if self.monitor.converged:
                # msg = 'Converged at iteration {} with current logprob {:.2f} and previous logprob {:.2f}'
                # cur_logprob = self.monitor.history[1] if len(self.monitor.history) > 1 else self.monitor.history[0]
                # prev_logprob = self.monitor.history[1] if len(self.monitor.history) > 1 else -1
                # msg = msg.format(it, cur_logprob, prev_logprob)
                # self.logger.debug(msg)
                break

        return self

    def __check_transcube(self, transcube):
        for o in range(self.n_obs):
            transmat = transcube[o,:,:]
            utils.assert_shape('transmat_d', (self.n_states, self.n_states), transmat.shape)
            sumframe = transmat.sum(axis=1)
            almost_one = np.isclose(sumframe, np.ones(self.n_states))
            error_rows = sumframe[np.invert(almost_one)]
            errmsg = 'transmat_d[{},:,:] does not sum to almost 1: {}'.format(o, error_rows)
            assert almost_one.all(), errmsg

            # col_sum = transmat.sum(axis=0)
            # col_ind = np.argwhere(col_sum == 0.).ravel()
            # errmsg = 'transmat_d[{},:,:] has columns that sum to zero'.format(o)
            # assert col_ind.shape[0] == 0, errmsg

    def __check_emitmat(self, emitmat):
        sumframe = emitmat.sum(axis=1)
        almost_one = np.isclose(sumframe, np.ones(self.n_states))
        error_rows = sumframe[np.invert(almost_one)]
        errmsg = 'emitmat_d does not sum to 1 to almost 1: {}'.format(error_rows)
        assert almost_one.all(), errmsg

        # col_sum = self.emitmat_d.sum(axis=0)
        # col_ind = np.argwhere(col_sum == 0.).ravel()
        # errmsg = 'emitmat_d has columns that sum to zero'
        # assert col_ind.shape[0] == 0, errmsg

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
            get0 = lambda a: a[0]
            get1 = lambda a: a[1]
            # to ensure that the resulting transcube still fulfill probability matrix requirement
            row_sum = stats['trans'].sum(axis=2)
            row_ind = np.argwhere(row_sum == 0.)
            ind0 = np.apply_along_axis(get0, 1, row_ind)
            ind1 = np.apply_along_axis(get1, 1, row_ind)
            stats['trans'][ind0,ind1,:] = self.transcube_d[ind0,ind1,:]

            # col_sum = stats['trans'].sum(axis=1)
            # col_ind = np.argwhere(col_sum == 0.).ravel()
            # ind0 = np.apply_along_axis(get0, 1, row_ind)
            # ind2 = np.apply_along_axis(get1, 1, row_ind)
            # stats['trans'][ind0,:,ind2] += 1e-4 # 0.01% of observing state

            # for o in range(self.n_obs):
            #     row_sum = stats['trans'][o,:,:].sum(axis=1)
            #     row_ind = np.argwhere(row_sum == 0.).ravel()
            #     stats['trans'][o,row_ind,:] = self.transcube_d[o,row_ind,:]

            #     # can overfit if EM samples does not contain some states so that the 
            #     # unobserved states has 0 over all states, i.e., never transitioned to
            #     # Avoid this by adding an epsilon probability
            #     col_sum = stats['trans'].sum(axis=0)
            #     col_ind = np.argwhere(col_sum == 0.).ravel()
            #     stats['trans'][o,:,col_ind] += 1e-4
            self.transcube_d = stats['trans'] # + self.transcube_d
            utils.normalize(self.transcube_d, axis=2)   # normalize row
            # self.__check_transcube(self.transcube_d)

        if 'o' in self.params:
            # if np.isnan(stats['obs']).any():
            #    raise ValueError('stats[obs] has nan: \n{}'.format(stats['obs']))
            # See the above explanation
            row_sum = stats['obs'].sum(axis=1)
            row_ind = np.argwhere(row_sum == 0.).ravel()
            stats['obs'][row_ind,:] = self.emitmat_d[row_ind,:]

            # Can overfit if EM samples does not contain some observations so that the 
            # unobserved observation has 0 over all states, i.e., never observable
            # Avoid this by adding an epsilon probability
            # col_sum = stats['obs'].sum(axis=0)
            # col_ind = np.argwhere(col_sum == 0.).ravel()
            # stats['obs'][:,col_ind] += 1e-4

            self.emitmat_d = stats['obs'] # + self.emitmat_d
            utils.normalize(self.emitmat_d, axis=1)
            # self.__check_emitmat(self.emitmat_d)

    def compute_distance_from_initstate(self, initstate, logfwd):
        work_buffer = logfwd.copy()
        denom = logsumexp(logfwd, axis=1)
        work_buffer = work_buffer - denom
        work_buffer = np.exp(work_buffer)
        utils.normalize(work_buffer, axis=1)

        outer = np.outer(initstate, work_buffer)
        # utils.assert_bounded('outer', outer, 0, 1)
        work_buffer = np.multiply(self.distmat, outer)
        dist = np.sum(work_buffer, axis=None)
        return dist

    def compute_expected_inc_distance(self, obs, logfwd, 
                                  prev_obs=None, prev_logfwd=None):
        """Compute expected distance between current state estimation derived from log forward probability and previous state estimation.

        :param obs int: observation
        :param logfwd array_like: log forward probability
        :param prev_obs int: previous observation
        :param prev_logfwd array_like: previous log forward probability
        :return: expected distance
        :rtype: float
        """
        if prev_obs is None:
            work_buffer = logfwd.copy()
            work_buffer = work_buffer - logsumexp(work_buffer, axis=1)
            logstartprob = utils.log_mask_zero(self.startprob).ravel()
            logstartprob = logstartprob[:,np.newaxis]
            logdistmat = utils.log_mask_zero(self.distmat)
            work_buffer = logdistmat + logstartprob + work_buffer
            work_buffer = logsumexp(work_buffer.ravel())
            return np.exp(work_buffer)

        arr_buffer = prev_logfwd.copy()
        utils.log_normalize(arr_buffer, axis=1) # to avoid underflow
        # P(Z_{t-1} | X_{1:t-1} = x_{1:t-1}), i.e., normalized forward probability at time t  - 1
        prev_stateprob = np.exp(arr_buffer)
        utils.normalize(prev_stateprob, axis=1) # to avoid not summing to 1 after exp

        # w_{i,j}(x_t)
        stateconf = self.conform_f(prev_stateprob, prev_obs, self.confmat, self.n_states)
        stateprob = _stateprob(prev_obs, stateconf, self.transcube, self.transcube_d)
        logstateprob = utils.log_mask_zero(stateprob)

        # v_{j}(x_{t+1})
        work_buffer = logstateprob.T + prev_logfwd
        cur_fwd_est = logsumexp(work_buffer, axis=1)
        cur_fwd_est = cur_fwd_est.reshape([1, self.n_states])
        # P(Z_t | X_{1:t} = x_{1:t}), i.e. normalized forward probability at time t
        cur_stateprob = cur_fwd_est.copy()
        utils.exp_log_normalize(cur_stateprob, axis=1)
        emitconf = self.conform_f(cur_stateprob, obs, self.confmat, self.n_states)
        obsprob = _emissionprob(obs, emitconf, self.emitmat, self.emitmat_d)
        logobsprob = utils.log_mask_zero(obsprob)[np.newaxis,:]
        prev_logfwd = prev_logfwd.ravel()[:,np.newaxis]

        # utils.assert_shape('logobsprob', (1, self.n_states), logobsprob.shape)
        # utils.assert_shape('logstateprob', (self.n_states, self.n_states), logstateprob.shape)
        # utils.assert_shape('prev_logfwd', (self.n_states, 1), prev_logfwd.shape)

        work_buffer = logobsprob + logstateprob + prev_logfwd - logsumexp(logfwd, axis=1)
        work_buffer = utils.log_mask_zero(self.distmat) + work_buffer
        log_dist = logsumexp(work_buffer.ravel())
        dist = np.exp(log_dist)

        return dist


"""Functions for parallelize EM fitting"""
def _emissionprob(obs, conf, emitmat, emitmat_d):
    """
    Computes P(x is obs at time t | z at time t) where x is the observation variable
    and z is the state variable. 

    :param obs int: observation at time t
    :param conf float: conformance between stateprob and obs
    """
    # logger.debug('conform: {}'.format(conf))
    # logger.debug('emitmat_d: \n{}'.format(emitmat_d[:,obs]))
    prob = conf * emitmat[:,obs] + (1 - conf) * emitmat_d[:,obs]
    return prob


def _stateprob(obs, conf, transcube, transcube_d):
    """
    Computes P(z at time t | z at time t - 1, x is obs at time t - 1) where x is the observation
    variable and z is the state variable.

    :param obs int: observed activity at time t - 1
    :param conf float: conformance between stateprob and obs
    """
    prob = conf * transcube[obs,:,:] + (1 - conf) * transcube_d[obs,:,:]
    # utils.assert_no_negatives('transcube[{},:,:]'.format(obs), transcube[obs,:,:])
    # utils.assert_no_negatives('transcube_d[{},:,:]'.format(obs), transcube_d[obs,:,:])
    # utils.assert_no_negatives('stateprob, conform: {}'.format(conf), prob)
    return prob


def _forward(n_states, transcube, transcube_d, emitmat, emitmat_d, confmat, 
             conform_f, obs, prev_obs=None, prev_fwd=None, startprob=None):
    """Computes the log forward probability.

    :param obs int: observation
    :param prev_obs int, optional: previous observation if any
    :param prev_fwd array_like, optional: previous log forward probability for all states
    :return: log forward probability, conformance array, log state probability, log emission probability
    """
    conf_arr = np.full(3, -1.)

    if prev_fwd is None:
        # logger.debug('startprob: {}'.format(startprob))
        emitconf = conform_f(startprob, obs, confmat, n_states)
        obsprob = _emissionprob(obs, emitconf, emitmat, emitmat_d)

        # n_nonzeros = np.count_nonzero(obsprob)
        # logger.info('Number of non-zeros (obsprob): {}'.format(n_nonzeros))
        # logger.info('Max obs prob: {} at {}'.format(obsprob.max(), np.argmax(obsprob)))
        
        logobsprob = utils.log_mask_zero(obsprob)[np.newaxis,:]
        logstartprob = utils.log_mask_zero(startprob)
        # utils.assert_shape('logobsprob', (1, n_states), logobsprob.shape)
        # utils.assert_shape('startprob', (1, n_states), logstartprob.shape)
        logfwd = logstartprob + logobsprob

        # check if it's a valid logfwd
        if logsumexp(logfwd, axis=1)[0] == -np.inf:
            msg = 'forward probability yielded 0 on replaying {}! ' \
                '\nlogobsprob: \n{} \nstartprob: \n{}'
            msg = msg.format(obs, logobsprob, logstartprob)
            warnings.warn(msg, category=UserWarning)

            if logsumexp(logobsprob, axis=1)[0] > -np.inf:
                logfwd = logobsprob.copy()
            else:
                logfwd = logstartprob.copy()
        
        fwd = logfwd.copy()
        utils.exp_log_normalize(fwd, axis=1)
        conf_arr[HMMConf.SECOND_IND] = emitconf[0]
        conf_arr[HMMConf.UPDATED_IND] = conform_f(fwd, obs, confmat, n_states)
        logstateprob = np.full((n_states, n_states), -np.inf) # zero everything
        return logfwd, conf_arr, logstateprob, logobsprob

    # P(Z_{t-1} | X_{1:t-1} = x_{1:t-1}), i.e., normalized forward probability at time t  - 1
    prev_stateprob = prev_fwd.copy()
    utils.exp_log_normalize(prev_stateprob, axis=1)

    stateconf = conform_f(prev_stateprob, prev_obs, confmat, n_states)
    stateprob = _stateprob(prev_obs, stateconf, transcube, transcube_d)
    logstateprob = utils.log_mask_zero(stateprob)

    # n_nonzeros = np.count_nonzero(stateprob)
    # logger.debug('No. non-zero vals: {}'.format(n_nonzeros))
    # logger.debug('Previous fwd: \n{}'.format(prev_fwd))
    # logger.debug('state prob: \n{}'.format(stateprob))
    # logger.debug('log state prob: \n{}'.format(logstateprob))

    work_buffer = logstateprob.T + prev_fwd
    # logger.debug('No. nan vals: {}'.format(np.count_nonzero(np.isnan(work_buffer))))
    cur_fwd_est = logsumexp(work_buffer, axis=1)
    cur_fwd_est = cur_fwd_est.reshape([1, n_states])

    # some helpful loggings during development...
    cur_stateprob = cur_fwd_est.copy()
    utils.exp_log_normalize(cur_stateprob, axis=1)

    # msg0 = '   Log state estimate of time t before observation at time t: \n{}'
    # msg1 = 'W. State estimate of time t before observation at time t: \n{}'
    # msg0 = msg0.format(cur_fwd_est)
    # msg1 = msg1.format(cur_stateprob)
    # logger.debug(msg0)
    # logger.debug(msg1)

    emitconf = conform_f(cur_stateprob, obs, confmat, n_states)
    obsprob = _emissionprob(obs, emitconf, emitmat, emitmat_d)[np.newaxis,:]
    logobsprob = utils.log_mask_zero(obsprob)

    # msg2 = 'Likelihood of observation at states time t: \n{}'.format(obsprob)
    # msg3 = 'Conformance between state and observation at time t ' \
    #         'before observation adjustment: {:.2f}'.format(emitconf[0])
    # logger.debug(msg2)
    # logger.debug(msg3)

    # logger.debug('Current forward est: \n{}'.format(cur_fwd_est))

    # utils.assert_shape('logobsprob', (1, n_states), logobsprob.shape)
    # utils.assert_shape('cur_fwd_est', (1, n_states), cur_fwd_est.shape)

    logfwd = logobsprob + cur_fwd_est

    # check if it's a valid logfwd
    if logsumexp(logfwd, axis=1)[0] == -np.inf:
        msg = 'forward probability yielded 0! ' \
            '\nlogobsprob: \n{} ' \
            '\ntransitioned prev_logfwd: \n{}'.format(logobsprob, cur_fwd_est)
        warnings.warn(msg, category=UserWarning)
        # trust the data if possible
        if logsumexp(logobsprob, axis=1)[0] > -np.inf:
            logfwd = logobsprob.copy()
        # otherwise trust the current forward estimation
        elif logsumexp(cur_fwd_est, axis=1)[0] > -np.inf:
            logfwd = cur_fwd_est.copy()
        else:
            logfwd[0,:] = np.log(1. / n_states)


    stateprob = logfwd.copy()
    utils.exp_log_normalize(stateprob, axis=1)

    # logger.debug('logfwd: \n{}'.format(logfwd))

    conf_arr[HMMConf.FIRST_IND] = stateconf[0]
    conf_arr[HMMConf.SECOND_IND] = emitconf[0]
    conf_arr[HMMConf.UPDATED_IND] = conform_f(stateprob, obs, confmat, n_states)

    return logfwd, conf_arr, logstateprob, logobsprob


def forward(n_states, transcube, transcube_d, emitmat, emitmat_d, confmat,
            obs, prev_obs=None, prev_fwd=None, startprob=None):
    """Computes the log forward probability.

    :param obs int: observation
    :param prev_obs int, optional: previous observation if any
    :param prev_fwd array_like, optional: previous log forward probability for all states
    :return: log forward probability, conformance array
    """
    logfwd, conf_arr, _, _ = _forward(n_states, transcube, transcube_d, emitmat, emitmat_d,
                                      confmat, obs, prev_obs, prev_fwd, startprob)
    return logfwd, conf_arr


def _do_forward_pass(x, n_states, transcube, transcube_d, 
                     emitmat, emitmat_d, confmat, startprob, conform_f):
    """computes the forward lattice containing the forward probability of a single sequence of
    observations.

    :param x: array of observations
    :type x: array_like (n_samples, 1)
    :return: log likelihood, the forward lattice, the conformance lattice, state-transition and observation lattice
    """
    n_samples = x.shape[0]
    fwdlattice = np.ndarray((n_samples, n_states))
    conflattice = np.ndarray((n_samples, 3))
    framelogstateprob = np.ndarray((n_samples, n_states, n_states))
    framelogobsprob = np.ndarray((n_samples, n_states))

    times = []

    # first observation
    obs = x[0,0]
    start = time.time()
    fwd, conf, _, logobsprob = _forward(n_states, transcube, transcube_d, emitmat, emitmat_d, confmat,
                                        conform_f, obs, startprob=startprob)
    end = time.time()
    times.append(end - start)
    fwdlattice[0] = fwd
    conflattice[0] = conf
    framelogstateprob[0] = -1.
    framelogobsprob[0] = logobsprob

    prev_obs = obs
    prev_fwd = fwd
    for i in range(1, n_samples):
        obs = x[i,0]
        start = time.time()
        fwd, conf, logstateprob, logobsprob = _forward(n_states, transcube, transcube_d, emitmat, emitmat_d,
                                                       confmat, conform_f, obs, prev_obs, prev_fwd)
        end = time.time()
        times.append(end - start)
        fwdlattice[i] = fwd
        conflattice[i] = conf
        framelogstateprob[i] = logstateprob
        framelogobsprob[i] = logobsprob

        prev_obs = obs
        prev_fwd = fwd

    # logger.debug('average forward probability time: {:.4f}s'.format(np.mean(times)))

    with np.errstate(under='ignore'):
        logprob = logsumexp(fwdlattice[-1])
        return logprob, fwdlattice, conflattice, framelogstateprob, framelogobsprob


def backward(emitmat, emitmat_d, transcube, transcube_d, obs, prev_obs, conf_arr, prev_bwd=None):
    """Computes the log backward probability.

    :param obs int: observation
    :param prev_obs int: previous observation
    :param conf_arr array_like: conformance vector computed for prev_obs using forward probability
    :param prev_bwd array_like, optional: previous log backward probability 
    :return: log backward probability
    """
    emitprob = _emissionprob(obs, conf_arr[1], emitmat, emitmat_d)
    stateprob = _stateprob(prev_obs, conf_arr[0], transcube, transcube_d)
    logemitprob = utils.log_mask_zero(emitprob)
    logstateprob = utils.log_mask_zero(stateprob)

    # no need to transpose logstateprob since we are broadcasting addition across the rows
    summed = logemitprob + logstateprob
    if prev_bwd is None:
        logbwd = logsumexp(logemitprob + logstateprob, axis=1)
    else:
        logbwd = logsumexp(logemitprob + logstateprob + prev_bwd, axis=1)

    # logger.debug('Backward probability: \n{}'.format(logbwd))

    return logbwd


def do_backward_pass(X, conflattice, n_states, emitmat, emitmat_d, transcube, transcube_d):
    """Computes the backward lattice containing the backward probability of a single sequence 
    of observations.

    :param X: array of observations
    :type X: array_like (n_samples, 1)
    :param conflattice: 
    :return: the backward lattice
    """
    n_samples = X.shape[0]
    bwdlattice = np.ndarray((n_samples, n_states))
    
    # last observation bwd(T) = 1. for all states
    bwdlattice[-1,:] = 0.
    
    obs = X[-1,0]
    prev_bwd = bwdlattice[-1]
    for i in range(n_samples - 2, -1, -1): # compute bwd(T - 1) to bwd(1)
        prev_obs = X[i,0]
        conf_arr = conflattice[i+1]
        bwd = backward(emitmat, emitmat_d, transcube, transcube_d, obs, prev_obs, conf_arr, prev_bwd)
        bwdlattice[i] = bwd

        obs = prev_obs
        prev_bwd = bwd

    return bwdlattice


def _compute_posteriors(fwdlattice, bwdlattice):
    """Posterior likelihood of states given data.

    :param fwdlattice array_like: log forward probability 
    :param bwdlattice array_like: log backward probability
    """
    log_gamma = fwdlattice + bwdlattice
    utils.log_normalize(log_gamma, axis=1)  # this prevents underflow
    with np.errstate(under='ignore'):
        return np.exp(log_gamma)


def _initialize_sufficient_statistics(start_shape, transcube_shape, emitmat_shape):
    """Initialize sufficient statistics.

    :return: sufficient statistics
    """
    stats = {'nobs': 0,
             'start': np.zeros(start_shape),
             'trans': np.zeros(transcube_shape),
             'obs': np.zeros(emitmat_shape)}
    return stats


def _accumulate_sufficient_statistics(stats, X, logstateprob, logobsprob, conflattice, 
                                      posteriors, fwdlattice, bwdlattice, 
                                      params, n_states, n_obs):
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

    if 's' in params:
        stats['start'] += posteriors[0]

    if 't' in params:
        n_samples = logobsprob.shape[0]

        if n_samples <= 1:
            return

        # utils.assert_shape('logstateprob', (n_samples, n_states, n_states), logstateprob.shape)
        # utils.assert_shape('logobsprob', (n_samples, n_states), logobsprob.shape)

        log_xi_sum = np.full((n_obs, n_states, n_states), -np.inf)

        for t in range(n_samples - 1):
            # skip events that are perfectly conforming
            if conflattice[t, HMMConf.UPDATED_IND] >= 1.:  
                continue

            o = X[t]    # to identify the state transition matrix to update
            for i in range(n_states):
                for j in range(n_states):
                    to_add = (fwdlattice[t, i] + logstateprob[t, i, j]
                                + logobsprob[t + 1, j] + bwdlattice[t + 1, j])
                    log_xi_sum[o, i, j] = np.logaddexp(log_xi_sum[o, i, j], to_add)

        denominator = logsumexp(log_xi_sum, axis=2)

        # utils.assert_shape('denominator', (n_obs, n_states), denominator.shape)

        for o in range(n_obs):
            denominator_o = denominator[o,:].ravel()[:,np.newaxis]
            # only update values if sample affected the state-transition between i and j for o
            to_update = denominator_o != -np.inf
            # log_xi_sum[o,:] = np.subtract(log_xi_sum[o,:], denominator[o,:], where=to_update)
            np.subtract(log_xi_sum[o,:,:], denominator_o, out=log_xi_sum[o,:,:], where=to_update)

        with np.errstate(under='ignore'):
            stats['trans'] += np.exp(log_xi_sum)

    if 'o' in params:
        n_samples = logobsprob.shape[0]

        xi_sum = np.zeros((n_states, n_obs))
        n_deviations = 0
        for t, symbol in enumerate(np.concatenate(X)):
            # skip if it's perfectly conforming
            if conflattice[t, HMMConf.UPDATED_IND] >= 1.:
                continue

            # logger.debug('No. of non-zeros: {}'.format(np.count_nonzero(posteriors[t])))
            # stats['obs'][:, symbol] += posteriors[t]
            xi_sum[:,symbol] += posteriors[t]
            n_deviations += 1

        # logger.debug('There were {} non-conforming events'.format(n_deviations))

        if n_deviations == 0:
            return

        # denominator = stats['obs'].sum(axis=1)[:, np.newaxis]
        denominator = xi_sum.sum(axis=1)[:,np.newaxis]
        # avoid zero division by replacement as 1.
        # denominator[denominator == 0.] = 1.
        # stats['obs'] /= denominator

        # only update matrix if the sample affected the emission probability
        to_update = denominator != 0.
        # np.divide(stats['obs'], denominator, out=stats['obs'], where=to_update)
        np.divide(xi_sum, denominator, out=xi_sum, where=to_update)

        # if np.isnan(xi_sum).any():
        #     raise ValueError('xi_sum has nan: \n{}'.format(xi_sum))

        stats['obs'] += xi_sum


def _fit_worker(args):
    conform_f = args['conform_f']
    params = args['params']
    n_obs = args['n_obs']
    n_states = args['n_states']
    transcube = args['transcube']
    transcube_d = args['transcube_d']
    emitmat = args['emitmat']
    emitmat_d = args['emitmat_d']
    confmat = args['confmat']
    startprob = args['startprob']
    X = args['X']
    lengths = args['lengths']

    cur_logprob = 0
    stats = _initialize_sufficient_statistics(n_states, 
                                                transcube.shape, 
                                                emitmat.shape)

    for i, j in utils.iter_from_X_lengths(X, lengths):
        fwd_results = _do_forward_pass(X[i:j], n_states, transcube, transcube_d,
                                        emitmat, emitmat_d, confmat, startprob, conform_f)
        logprob = fwd_results[0]
        fwdlattice = fwd_results[1]
        conflattice = fwd_results[2]
        logstateprob = fwd_results[3]
        logobsprob = fwd_results[4]

        cur_logprob += logprob
        bwdlattice = do_backward_pass(X[i:j], conflattice, n_states, 
                                      emitmat, emitmat_d, transcube, transcube_d)
        posteriors = _compute_posteriors(fwdlattice, bwdlattice)

        _accumulate_sufficient_statistics(stats, X[i:j], logstateprob, logobsprob, 
                                            conflattice, posteriors, fwdlattice, bwdlattice, 
                                            params, n_states, n_obs)

    return cur_logprob, stats
