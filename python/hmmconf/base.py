import numpy as np
import multiprocessing as mp
from numba import njit, prange

from hmmconf.conform import *
from hmmconf.base_utils import *
from hmmconf.numba_utils import *
from hmmconf.utils import *

import scipy


logger = make_logger(__file__)


__all__ = [
    'compute_logemissionprob',
    'compute_logstateprob',
    'initialize_sufficient_statistics',
    'compute_logfwd',
    'compute_logbwd',
    'do_logfwd_pass',
    'do_logbwd_pass',
    'compute_posteriors',
    'accumulate_transcube',
    'accumulate_sufficient_statistics',
    'fit_singleprocess',
    'fit_multiprocess',
    'fit_worker',
    'do_mstep'
]


EXCEPTION_LOGFWD_SUM_TO_ZERO = 0


@njit('f8[:](u8, f8, f8[:,:], f8[:,:])')
def compute_logemissionprob(obs, conf, logemitmat, logemitmat_d):
    """
    Computes P(x is obs at time t | z at time t) where x is the observation variable
    and z is the state variable:

    conf * emitmat[:,obs] + (1 - conf) * emitmat_d[:,obs]

    :param obs int: observation at time t
    :param conf float: conformance between stateprob and obs
    """
    logconf = np.log(conf)
    loginvconf = np.log(1 - conf)
    logprob0 = logconf + logemitmat[:,obs]
    logprob1 = loginvconf + logemitmat_d[:,obs]
    return np.logaddexp(logprob0, logprob1)


@njit('f8[:,:](u8, f8, f8[:,:,:], f8[:,:,:])')
def compute_logstateprob(obs, conf, logtranscube, logtranscube_d):
    """
    Computes P(z at time t | z at time t - 1, x is obs at time t - 1) where x is the observation
    variable and z is the state variable:
     
    conf * transcube[obs,:,:] + (1 - conf) * transcube_d[obs,:,:]

    :param obs int: observed activity at time t - 1
    :param conf float: conformance between stateprob and obs
    """
    logconf = np.log(conf)
    loginvconf = np.log(1 - conf)
    logprob0 = logconf + logtranscube[obs,:,:]
    logprob1 = loginvconf + logtranscube_d[obs,:,:]

    # print('logconf: {}'.format(logconf))
    # print('logprob0: {}'.format(logprob0))
    res = np.logaddexp(logprob0, logprob1)

    assert res.ndim == 2
    n_states = logtranscube.shape[1]
    assert res.shape == (n_states, n_states)

    return res

@njit(parallel=True)
def compute_logfwd(logtranscube, logtranscube_d, logemitmat, logemitmat_d,
                   confmat, obs, prev_obs=None, prev_logfwd=None, logstartprob=None):
    """Computes the log forward probability.

    :param obs int: observation
    :param prev_obs int, optional: previous observation if any
    :param prev_fwd array_like, optional: previous log forward probability for all states
    :return: log forward probability, conformance array, log state probability, log emission probability
    """
    n_obs = logtranscube.shape[0]
    n_states = logtranscube.shape[1]
    # conformance values
    emitconf = -1.
    stateconf = -1.
    finalconf = -1.
    exception = None

    if prev_logfwd is None:
#        print('logstartprob shape: {}'.format(logstartprob.shape))
#        print('confmat shape: {}'.format(confmat.shape))
#        logger.debug('logstartprob shape: {}'.format(logstartprob.shape))
#        logger.debug('confmat shape: {}'.format(confmat.shape))
#        sum_ = np.exp(logsumexp1d(logstartprob))[0]
#        logger.debug('startprob sum: {}'.format(sum_))

        emitconf = logconform(logstartprob, obs, confmat)
        logobsprob = compute_logemissionprob(obs, emitconf, logemitmat, logemitmat_d)
        
        # logfwd does not have to sum to 1!
        logfwd = logstartprob + logobsprob

        # check for validity
#        assert logfwd.ndim == 1
#        assert logfwd.shape == (n_states,)
        logfwd_sum = logsumexp1d(logfwd)[0]
#        logger.debug('logfwd_sum: {}'.format(logfwd_sum))
        if np.isinf(logfwd_sum):
#            err_msg = 'Forward probability yielded 0 on replaying {}'.format(obs)
#            print(err_msg)
            exception = EXCEPTION_LOGFWD_SUM_TO_ZERO
            logfwd[:] = np.log(1. / n_states)

#        sum_ = np.exp(logsumexp1d(logfwd))[0]
#        logger.debug('logfwd sum: {}'.format(sum_))

        # get the Petri net marking log probability vector
        logmarkingprob = log_normalize1d(logfwd, inplace=False)
        finalconf = logconform(logmarkingprob, obs, confmat)
        # zero everything
        logstateprob = np.full((n_states, n_states), -np.inf)

        return logfwd, emitconf, stateconf, finalconf, logstateprob, logobsprob, exception

    # P(Z_{t-1} | X_{1:t-1} = x_{1:t-1}), i.e., normalized forward probability at time t - 1
    # get the Petri net marking log probability vector
    logmarkingprob = log_normalize1d(prev_logfwd, inplace=False)
    stateconf = logconform(logmarkingprob, prev_obs, confmat)
    logstateprob = compute_logstateprob(prev_obs, stateconf, logtranscube, logtranscube_d)

    work_buffer = logstateprob.T
    for i in prange(n_states):
        work_buffer[i,:] += prev_logfwd
    # work_buffer = logstateprob.T + prev_logfwd
    cur_logfwd_est = logsumexp2d(work_buffer, axis=1)

#    assert cur_logfwd_est.ndim == 1
#    assert cur_logfwd_est.shape[0] == n_states

    # get the Petri net marking log probability vector
    logmarkingprob = log_normalize1d(cur_logfwd_est, inplace=False)
    emitconf = logconform(logmarkingprob, obs, confmat)
    
#    # Used in the identification of the zero division bug for logsumexp1d function
#    if np.isnan(emitconf):
#        logger.debug('Emitconf is nan!')
#        logger.debug('prev_logfwd: \n{}'.format(prev_logfwd))
#        logger.debug('logstateprob: \n{}'.format(logstateprob))
#        is_all_zeros = np.isinf(logstateprob)
#        logger.debug('logstateprob all zeros: {}'.format(is_all_zeros.all()))
#        is_finite = np.isfinite(logstateprob)
#        non_zeros = logstateprob[is_finite]
#        logger.debug('logstateprob non-zero: \n{}'.format(non_zeros))
#        logger.debug('cur_logfwd_est: \n{}'.format(cur_logfwd_est))
#        logger.debug('logmarkingprob: \n{}'.format(logmarkingprob))
#        raise ValueError

    logobsprob = compute_logemissionprob(obs, emitconf, logemitmat, logemitmat_d)

#    assert logobsprob.ndim == 1
#    assert logobsprob.shape == (n_states,)

    logfwd = logobsprob + cur_logfwd_est

    # check for validity
#    assert logfwd.ndim == 1
#    assert logfwd.shape == (n_states,)
    logfwd_sum = logsumexp1d(logfwd)[0]
    # logger.debug('logfwd_sum: {}'.format(logfwd_sum))
    if np.isinf(logfwd_sum):
        # err_msg = 'Forward probability yielded 0 on replaying {}'.format(obs)
        # print(err_msg)
        exception = EXCEPTION_LOGFWD_SUM_TO_ZERO
        logfwd[:] = np.log(1. / n_states)

    # get the Petri net marking log probability vector
    logmarkingprob = log_normalize1d(logfwd, inplace=False)
    finalconf = logconform(logmarkingprob, obs, confmat)

    return logfwd, emitconf, stateconf, finalconf, logstateprob, logobsprob, exception


@njit
def compute_logbwd(logemitmat, logemitmat_d, logtranscube, logtranscube_d, 
                   obs, prev_obs, emitconf, stateconf, prev_logbwd=None):
    """Computes the log backward probability.

    :param obs int: observation
    :param prev_obs int: previous observation
    :param emitconf float: emission conformance
    :param stateconf float: state transition conformance
    :param prev_logbwd array_like, optional: previous log backward probability 
    :return: log backward probability
    """
    logobsprob = compute_logemissionprob(obs, stateconf, logemitmat, logemitmat_d)
    logstateprob = compute_logstateprob(prev_obs, emitconf, logtranscube, logtranscube_d)

    sum_ = logobsprob + logstateprob
    if prev_logbwd is None:
        logbwd = logsumexp2d(sum_, axis=1)
    else:
        sum_ = sum_ + prev_logbwd
        logbwd = logsumexp2d(sum_, axis=1) 

    return logbwd


@njit
def do_logfwd_pass(X, logtranscube, logtranscube_d, logemitmat, logemitmat_d,
                   confmat, logstartprob):
    """computes the forward lattice containing the forward probability of a single sequence of
    observations.

    :param x: array of observations
    :type x: array_like (n_samples, 1)
    :return: log likelihood, the forward lattice, the conformance lattice, state-transition and observation lattice
    """
    n_samples = X.shape[0]
    n_states = logtranscube.shape[1]

    logfwdlattice = np.empty((n_samples, n_states))
    emitconf_arr = np.empty(n_samples)
    stateconf_arr = np.empty(n_samples)
    finalconf_arr = np.empty(n_samples)
    framelogstateprob = np.empty((n_samples, n_states, n_states))
    framelogobsprob = np.empty((n_samples, n_states))

    # first observation
    obs = X[0,0]
    result = compute_logfwd(logtranscube, logtranscube_d, logemitmat, logemitmat_d,
                            confmat, obs, logstartprob=logstartprob)

    logfwd = result[0]
    emitconf = result[1]
    stateconf = result[2]
    finalconf = result[3]
    logstateprob = result[4]
    logobsprob = result[5]

    # add to result
    logfwdlattice[0] = logfwd
    emitconf_arr[0] = emitconf
    stateconf_arr[0] = stateconf
    finalconf_arr[0] = finalconf
    framelogstateprob[0] = -1.
    framelogobsprob[0] = logobsprob

    prev_obs = obs
    prev_logfwd = logfwd

    for i in range(1, n_samples):
        obs = X[i,0]
        result = compute_logfwd(logtranscube, logtranscube_d, logemitmat, logemitmat_d,
                                confmat, obs, prev_obs=prev_obs, prev_logfwd=prev_logfwd, 
                                logstartprob=logstartprob)

        logfwd = result[0]
        emitconf = result[1]
        stateconf = result[2]
        finalconf = result[3]
        logstateprob = result[4]
        logobsprob = result[5]

        # add to result
        logfwdlattice[i] = logfwd
        emitconf_arr[i] = emitconf
        stateconf_arr[i] = stateconf
        finalconf_arr[i] = finalconf
        framelogstateprob[i] = logstateprob
        framelogobsprob[i] = logobsprob

        prev_obs = obs
        prev_logfwd = logfwd

    return logfwdlattice, emitconf_arr, stateconf_arr, finalconf_arr, framelogstateprob, framelogobsprob


@njit
def do_logbwd_pass(X, emitconf_arr, stateconf_arr, logemitmat, 
                   logemitmat_d, logtranscube, logtranscube_d):
    """Computes the backward lattice containing the backward log probability of a single sequence 
    of observations.

    :param X: array of observations
    :type X: array_like (n_samples, 1)
    :return: the backward lattice
    """
    n_samples = X.shape[0]
    n_states = logtranscube.shape[1]

    logbwdlattice = np.empty((n_samples, n_states))

    # last observation bwd(T) = 1 for all states
    logbwdlattice[-1,:] = 0

    obs = X[-1,0]
    prev_logbwd = logbwdlattice[-1]

    for i in range(n_samples - 2, -1, -1): # compute logbwd(T - 1) to logbwd(1)
        prev_obs = X[i,0]
        emitconf = emitconf_arr[i+1]
        stateconf = stateconf_arr[i+1]
        logbwd = compute_logbwd(logemitmat, logemitmat_d, logtranscube, logtranscube_d,
                                obs, prev_obs, emitconf, stateconf, prev_logbwd)
        logbwdlattice[i] = logbwd

        obs = prev_obs
        prev_logbwd = logbwd

    return logbwdlattice


def compute_posteriors(logfwdlattice, logbwdlattice):
    """Posterior likelihood of states given data.

    :param fwdlattice array_like: log forward probability 
    :param bwdlattice array_like: log backward probability
    """
    log_gamma = logfwdlattice + logbwdlattice
    log_normalize(log_gamma, axis=1)  # this prevents underflow
    with np.errstate(under='ignore'):
        return np.exp(log_gamma)


# @njit('Tuple((i8[:], f8[:], f8[:,:,:], f8[:,:]))(u8, u8)')
@njit
def initialize_sufficient_statistics(n_obs, n_states):
    # four statistics to keep track of 
    nobs = np.zeros(1, dtype=np.int64)
    startprob = np.zeros(n_states)
    c_trans_log_numerator = np.zeros((n_obs, n_states, n_states))
    nc_trans_log_numerator = np.zeros((n_obs, n_states, n_states))
    c_obs_numerator = np.zeros((n_states, n_obs))
    nc_obs_numerator = np.zeros((n_states, n_obs))
    return nobs, startprob, c_trans_log_numerator, nc_trans_log_numerator, c_obs_numerator, nc_obs_numerator


# @njit('(f8[:], f8[:,:], f8[:,:,:], f8[:,:], f8[:,:], f8[:,:,:], i8[:])', parallel=True)
@njit(parallel=True)
def accumulate_transcube(finalconf_arr, logfwdlattice, framelogstateprob, framelogobsprob, logbwdlattice, 
                         c_log_xi_sum, nc_log_xi_sum, X, 
                         conf_tol=1e-8, update_conform=True, update_nconform=True):
    if not update_conform and not update_nconform:
        return 

    n_samples = framelogobsprob.shape[0]
    n_states = nc_log_xi_sum.shape[1]
    n_obs = nc_log_xi_sum.shape[0]

    for t in prange(n_samples - 1):
        o = X[t]    # to identify the state transition matrix to update
        finalconf = finalconf_arr[t]
        # decide whether if it is conforming, taking into account slight imprecision using tolerance
        is_conf = isclose(finalconf, 1, conf_tol)

        for i in prange(n_states):
            for j in prange(n_states):
                to_add = (logfwdlattice[t, i] + framelogstateprob[t, i, j] 
                        + framelogobsprob[t + 1, j] + logbwdlattice[t + 1, j])

                if update_conform and is_conf:
                    c_log_xi_sum[o, i, j] = np.logaddexp(c_log_xi_sum[o, i, j], to_add)

                if update_nconform and not is_conf:
                    nc_log_xi_sum[o, i, j] = np.logaddexp(nc_log_xi_sum[o, i, j], to_add)


# @njit('(i8[:], u8[:], f8[:,:,:], f8[:,:], i8[:], f8[:,:,:], f8[:,:], f8[:], f8[:,:], f8[:,:], f8[:,:], b1[:], u8, u8)')
def accumulate_sufficient_statistics(nobs, startprob, 
                                     c_trans_log_numerator, nc_trans_log_numerator, 
                                     c_obs_numerator, nc_obs_numerator,
                                     X, framelogstateprob, framelogobsprob,
                                     finalconf_arr, posteriors, logfwdlattice,
                                     logbwdlattice, params, n_obs, n_states, 
                                     conf_tol=1e-8):
    """Updates sufficient statistics from a given sample.

    :param stats dict: dictionary storing the sufficient statistics of the HMM 
    :param logstateprob array_like: Log state probability at each time frame 1 to T
    :param logobsprob array_like: Log observation probability at each time frame 1 to T
    :param conflattice array_like: Conformance at each time frame 1 to T
    :param posteriors array_like: Posterior likelihood at each time frame 1 to T
    :param fwdlattice array_like: Log forward probability at each time frame 1 to T
    :param bwdlattice array_like: Log backward probability at each time frame 1 to T
    
    # note that params is a boolean array!!
    """
    nobs[0] += 1

    if params[PARAM_START]:
        startprob[:] = startprob[:] + posteriors[0, :]

    trans_update_conform = params[PARAM_CONFORM_TRANS]
    trans_update_nconform = params[PARAM_NCONFORM_TRANS]

    if trans_update_conform or trans_update_nconform:
        n_samples = framelogobsprob.shape[0]

        if n_samples <= 1:
            return

        c_log_xi_sum = np.full((n_obs, n_states, n_states), -np.inf)
        nc_log_xi_sum = np.full((n_obs, n_states, n_states), -np.inf)

        accumulate_transcube(finalconf_arr, logfwdlattice, framelogstateprob, 
                             framelogobsprob, logbwdlattice, 
                             c_log_xi_sum, nc_log_xi_sum, X, 
                             conf_tol=conf_tol, 
                             update_conform=trans_update_conform,
                             update_nconform=trans_update_nconform)

        c_trans_log_numerator[:,:,:] = np.logaddexp(c_trans_log_numerator, c_log_xi_sum)
        nc_trans_log_numerator[:,:,:] = np.logaddexp(nc_trans_log_numerator, nc_log_xi_sum)

    obs_update_conform = params[PARAM_CONFORM_OBS]
    obs_update_nconform = params[PARAM_NCONFORM_OBS]

    if obs_update_conform or obs_update_nconform:
        n_samples = framelogobsprob.shape[0]
        
        c_xi_sum = np.zeros((n_states, n_obs))
        nc_xi_sum = np.zeros((n_states, n_obs))

        n_deviations = 0

        for t, symbol in enumerate(np.concatenate(X)):
            finalconf = finalconf_arr[t]
            is_conf = isclose(finalconf, 1, conf_tol)
            n_deviations += int(is_conf)

            if obs_update_conform and is_conf:
                c_xi_sum[:, symbol] += posteriors[t]
            if obs_update_nconform and not is_conf:
                nc_xi_sum[:, symbol] += posteriors[t]

        c_obs_numerator[:,:] += c_xi_sum
        nc_obs_numerator[:,:] += nc_xi_sum


def fit_singleprocess(X, lengths, params, logtranscube, logtranscube_d, 
                     logemitmat, logemitmat_d, confmat, logstartprob, conf_tol):
    args = {
        'params': params,
        'logtranscube': logtranscube,
        'logtranscube_d': logtranscube_d,
        'logemitmat': logemitmat,
        'logemitmat_d': logemitmat_d,
        'confmat': confmat,
        'logstartprob': logstartprob,
        'X': X,
        'lengths': lengths,
        'conf_tol': conf_tol
    }
    
    logprob, stats = fit_worker(args)
    results = [(logprob, stats)]
    return results


def fit_multiprocess(X, lengths, params, logtranscube, logtranscube_d, 
                     logemitmat, logemitmat_d, confmat, logstartprob, conf_tol, n_procs=-1):
    n_procs = n_procs if n_procs > 0 and n_procs <= mp.cpu_count() else mp.cpu_count()
    X_parts, lengths_parts = partition_X(X, lengths, n_procs)
    n_procs = len(X_parts)

    args_list = [{
        'params': params,
        'logtranscube': logtranscube,
        'logtranscube_d': logtranscube_d,
        'logemitmat': logemitmat,
        'logemitmat_d': logemitmat_d,
        'confmat': confmat,
        'logstartprob': logstartprob,
        'X': X_parts[i],
        'lengths': lengths_parts[i],
        'conf_tol': conf_tol
    } for i in range(n_procs)]

    if n_procs == 1:
        logprob, stats = fit_worker(args_list[0])
        results = [(logprob, stats)]
    else:
        pool = mp.Pool(processes=n_procs)
        results = pool.map(fit_worker, args_list)
        pool.close()

    return results


def fit_worker(args):
    params = args['params']
    logtranscube = args['logtranscube']
    logtranscube_d = args['logtranscube_d']
    logemitmat = args['logemitmat']
    logemitmat_d = args['logemitmat_d']
    confmat = args['confmat']
    logstartprob = args['logstartprob']
    X = args['X']
    lengths = args['lengths']
    conf_tol = args['conf_tol']

    n_obs = logtranscube.shape[0]
    n_states = logtranscube.shape[1]
    cur_logprob = 0
    stats = initialize_sufficient_statistics(n_obs, n_states)

    nobs = stats[STATS_NOBS]
    startprob = stats[STATS_STARTPROB]
    c_trans_log_numerator = stats[STATS_C_TRANS_LOG_NUMERATOR]
    c_obs_numerator = stats[STATS_C_OBS_NUMERATOR]
    nc_trans_log_numerator = stats[STATS_NC_TRANS_LOG_NUMERATOR]
    nc_obs_numerator = stats[STATS_NC_OBS_NUMERATOR]

    for i, j in iter_from_X_lengths(X, lengths):
        logfwd_results = do_logfwd_pass(X[i:j], logtranscube, logtranscube_d,
                                        logemitmat, logemitmat_d, confmat, logstartprob)

        logfwdlattice = logfwd_results[0]
        emitconf_arr = logfwd_results[1]
        stateconf_arr = logfwd_results[2]
        finalconf_arr = logfwd_results[3]
        framelogstateprob = logfwd_results[4]
        framelogobsprob = logfwd_results[5]

        with np.errstate(under='ignore'):
            logprob = scipy.special.logsumexp(logfwdlattice[-1])

        cur_logprob += logprob
        logbwdlattice = do_logbwd_pass(X[i:j], emitconf_arr, stateconf_arr, 
                                       logemitmat, logemitmat_d, logtranscube, logtranscube_d)
        posteriors = compute_posteriors(logfwdlattice, logbwdlattice)
        accumulate_sufficient_statistics(nobs, startprob, 
                                         c_trans_log_numerator, nc_trans_log_numerator,
                                         c_obs_numerator, nc_obs_numerator,
                                         X[i:j], framelogstateprob, framelogobsprob,
                                         finalconf_arr, posteriors, logfwdlattice, logbwdlattice,
                                         params, n_obs, n_states, conf_tol)

    return cur_logprob, (nobs, startprob, c_trans_log_numerator, nc_trans_log_numerator, c_obs_numerator, nc_obs_numerator)


def do_mstep_transcube(logtranscube, trans_log_numerator):
    n_obs = logtranscube.shape[0]
    trans_log_denominator = scipy.special.logsumexp(trans_log_numerator, axis=2)

    for o in range(n_obs):
        log_denominator_o = trans_log_denominator[o,:].ravel()[:,np.newaxis]
        to_update = log_denominator_o != -np.inf
        np.subtract(trans_log_numerator[o,:,:], log_denominator_o, 
                    out=trans_log_numerator[o,:,:], where=to_update)

    # ensure that the resulting transcube still fulfill probability matrix requirements
    get0 = lambda a: a[0]
    get1 = lambda a: a[1]

    row_logsumexp = scipy.special.logsumexp(trans_log_numerator, axis=2)
    row_ind = np.argwhere(row_logsumexp == -np.inf)

    # some rows have all zeros in trans_log_numerator
    # use the values from the old logtranscube
    if row_ind.shape[0] > 0:
        ind0 = np.apply_along_axis(get0, 1, row_ind)
        ing1 = np.apply_along_axis(get1, 1, row_ind)
        trans_log_numerator[ind0, ind1, :] = logtranscube[ind0, ind1, :]

    # update transcube_d
    logtranscube[:,:,:] = trans_log_numerator[:,:,:]
    log_normalize(logtranscube, axis=2)


def do_mstep_emitmat(logemitmat, obs_numerator):
    obs_denominator = obs_numerator.sum(axis=1)[:, np.newaxis]
    to_update = obs_denominator != 0.
    np.divide(obs_numerator, obs_denominator, out=obs_numerator, where=to_update)
    
    row_sum = obs_numerator.sum(axis=1)
    row_ind = np.argwhere(row_sum == 0.).ravel()
    emitmat = np.exp(logemitmat)
    if row_ind.shape[0] > 0:
        obs_numerator[row_ind,:] = emitmat[row_ind,:]
    nonzero = obs_numerator != 0
    logobs = np.log(obs_numerator, where=nonzero)
    logobs[obs_numerator == 0] = -np.inf
    logemitmat[:,:] = logobs[:,:]
    log_normalize(logemitmat, axis=1)


def do_mstep(stats, params, logstartprob, logtranscube, logtranscube_d, logemitmat, logemitmat_d):
    if params[PARAM_START]:
        startprob = stats[1]
        logstartprob[:] = logstartprob + np.log(startprob)
        log_normalize(logstartprob, axis=1)

    trans_update_conform = params[PARAM_CONFORM_TRANS]
    trans_update_nconform = params[PARAM_NCONFORM_TRANS]
    obs_update_conform = params[PARAM_CONFORM_OBS]
    obs_update_nconform = params[PARAM_NCONFORM_OBS]

    if trans_update_conform:
        c_trans_log_numerator = stats[STATS_C_TRANS_LOG_NUMERATOR]
        do_mstep_transcube(logtranscube, c_trans_log_numerator)

    if trans_update_nconform:
        nc_trans_log_numerator = stats[STATS_NC_TRANS_LOG_NUMERATOR]
        do_mstep_transcube(logtranscube_d, nc_trans_log_numerator)

    if obs_update_conform:
        c_obs_numerator = stats[STATS_C_OBS_NUMERATOR]
        do_mstep_emitmat(logemitmat, c_obs_numerator)

    if obs_update_nconform:
        nc_obs_numerator = stats[STATS_NC_OBS_NUMERATOR]
        do_mstep_emitmat(logemitmat_d, nc_obs_numerator)
