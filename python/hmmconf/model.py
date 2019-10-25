import numpy as np

from hmmconf.base_utils import *
from hmmconf.utils import *
from hmmconf.base import *


__all__ = [
    'HMMConf'
]


class HMMConf:
    def __init__(self, startprob, transcube, emitmat, confmat, 
                 int2state, int2obs, params='to', conf_tol=1e-8, n_iter=10,
                 tol=1e-2, verbose=False, n_procs=None, random_seed=123):
        assert_no_negatives('transcube', transcube)
        assert_no_negatives('emitmat', emitmat)
        n_obs, n_states = transcube.shape[0], transcube.shape[1]
        assert_shape('Emitmat', (n_states, n_obs), emitmat.shape)
        assert_shape('Confmat', (n_obs, n_states), confmat.shape)

        self.logger = make_logger(HMMConf.__name__)

        self.logger.info('Transcube shape: {}'.format(transcube.shape))
        self.logger.info('Emitmat shape: {}'.format(emitmat.shape))
        self.logger.info('Confmat shape: {}'.format(confmat.shape))

        self.params = np.full(N_PARAMS, False, dtype=np.bool)
        self.setup_params(params)

        self.random_seed = random_seed
        self.logstartprob = np.log(startprob)
        self.logtranscube = np.log(transcube)
        self.logemitmat = np.log(emitmat)
        log_normalize(self.logstartprob)
        log_normalize(self.logtranscube, axis=2)
        log_normalize(self.logemitmat, axis=1)

        # self.logtranscube_d = np.zeros(transcube.shape) + 1. / transcube.shape[1]
        # self.logtranscube_d = np.log(self.logtranscube_d)
        # self.logemitmat_d = np.zeros(emitmat.shape) + 1. / emitmat.shape[1]
        # self.logemitmat_d = np.log(self.logemitmat_d)
        np.random.seed(random_seed)
        self.logtranscube_d = np.random.rand(*transcube.shape)
        normalize(self.logtranscube_d, axis=2)
        self.logtranscube_d = np.log(self.logtranscube_d)

        self.logemitmat_d = np.random.rand(*emitmat.shape)
        normalize(self.logemitmat_d, axis=1)
        self.logemitmat_d = np.log(self.logemitmat_d)

        self.confmat = confmat
        self.int2state = int2state
        self.int2obs = int2obs
        self.conf_tol = conf_tol
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.monitor = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)
        self.n_procs = n_procs
        self.n_exceptions = 0

    def setup_params(self, params):
        if 't' in params:
            self.params[PARAM_NCONFORM_TRANS] = True
        if 'o' in params:
            self.params[PARAM_NCONFORM_OBS] = True
        if 'a' in params:
            self.params[PARAM_CONFORM_TRANS] = True
        if 'b' in params:
            self.params[PARAM_CONFORM_OBS] = True
        if 's' in params:
            self.params[PARAM_START] = True

    def compute_logfwd(self, obs, prev_obs=None, prev_logfwd=None):
        """Computes the log forward probability.

        :param obs int: observation
        :param prev_obs int, optional: previous observation if any
        :param prev_fwd array_like, optional: previous log forward probability for all states
        :return: log forward probability, conformance array
        """
        results = compute_logfwd(
            self.logtranscube, self.logtranscube_d,
            self.logemitmat, self.logemitmat_d,
            self.confmat, obs, prev_obs, prev_logfwd, self.logstartprob
        )

        logfwd = results[0]
        emitconf = results[1]
        stateconf = results[2]
        finalconf = results[3]
        logstateprob = results[4]
        logobsprob = results[5]
        exception = results[6] 

        return logfwd, emitconf, stateconf, finalconf, exception

    def fit(self, X, lengths):
        """Estimate model parameters using EM.

        :param X array_like, shape (n_samples, 1): sample data
        :param lengths array_like, shape (n_sequences,): lengths of the individual sequences in ``X``. The sum of these should be ``n_samples``.
        :return: ``self``
        """
        self.monitor._reset()
        n_obs = self.logtranscube.shape[0]
        n_states = self.logtranscube.shape[1]

        for it in range(self.n_iter):
            cur_logprob = 0
            stats = initialize_sufficient_statistics(n_obs, n_states)
            startprob = stats[STATS_STARTPROB]
            nobs = stats[STATS_NOBS]
            c_trans_log_numerator = stats[STATS_C_TRANS_LOG_NUMERATOR]
            nc_trans_log_numerator = stats[STATS_NC_TRANS_LOG_NUMERATOR]
            c_obs_numerator = stats[STATS_C_OBS_NUMERATOR]
            nc_obs_numerator = stats[STATS_NC_OBS_NUMERATOR]

            if self.n_procs is None or self.n_procs == 1:
                results = fit_singleprocess(X, lengths, self.params,
                                            self.logtranscube, self.logtranscube_d,
                                            self.logemitmat, self.logemitmat_d, 
                                            self.confmat, self.logstartprob, self.conf_tol)
            else:
                results = fit_multiprocess(X, lengths, self.params,
                                           self.logtranscube, self.logtranscube_d,
                                           self.logemitmat, self.logemitmat_d,
                                           self.confmat, self.logstartprob, self.conf_tol, self.n_procs)

            for i in range(len(results)):
                logprob_i, stats_i = results[i]
                startprob_i = stats_i[STATS_STARTPROB]
                nobs_i = stats_i[STATS_NOBS]
                c_trans_log_numerator_i = stats_i[STATS_C_TRANS_LOG_NUMERATOR]
                nc_trans_log_numerator_i = stats_i[STATS_NC_TRANS_LOG_NUMERATOR]
                c_obs_numerator_i = stats_i[STATS_C_OBS_NUMERATOR]
                nc_obs_numerator_i = stats_i[STATS_NC_OBS_NUMERATOR]

                np.logaddexp(c_trans_log_numerator, c_trans_log_numerator_i, out=c_trans_log_numerator)
                np.logaddexp(nc_trans_log_numerator, nc_trans_log_numerator_i, out=nc_trans_log_numerator)

                np.add(c_obs_numerator, c_obs_numerator_i, out=c_obs_numerator)
                np.add(nc_obs_numerator, nc_obs_numerator_i, out=nc_obs_numerator)

                np.add(startprob, startprob_i, startprob)
                np.add(nobs, nobs_i, nobs)
                cur_logprob += logprob_i

            # has_finite = np.isfinite(stats[STATS_C_TRANS_LOG_NUMERATOR]).any()
            # self.logger.info('trans_log_numerator has non-zero values: {}'.format(has_finite))
            # has_finite = np.isfinite(stats[STATS_C_OBS_NUMERATOR]).any()
            # self.logger.info('obs_numerator has non-zero values: {}'.format(has_finite))

            # work_buffer = self.logtranscube_d.copy()
            # work_buffer1 = self.logemitmat.copy()

            do_mstep(stats, self.params, self.logstartprob, 
                     self.logtranscube, self.logtranscube_d,
                     self.logemitmat, self.logemitmat_d)

            # diff = self.logtranscube_d - work_buffer
            # has_diff = diff != 0
            # total_diff = diff.sum()
            # uniq_diff = np.unique(self.logtranscube_d[has_diff].ravel())
            # self.logger.info('logtranscube_d has diff: {}'.format(has_diff.any()))
            # self.logger.info('logtranscube_d total diff: {:.5f}'.format(total_diff))
            # self.logger.info('logtranscube_d unique diff: {}'.format(uniq_diff))
            # diff = self.logemitmat - work_buffer1
            # has_diff = diff != 0
            # self.logger.info('logemitmat has diff: {}'.format(has_diff.any()))

            # trans_has_nan = np.isnan(self.logtranscube).any()
            # emit_has_nan = np.isnan(self.logemitmat).any()
            # self.logger.info('logtranscube has nan: {}'.format(trans_has_nan))
            # self.logger.info('logemitmat has nan: {}'.format(emit_has_nan))

            # assert_emitmat_validity(self.logemitmat, n_obs, n_states)
            # assert_transcube_validity(self.logtranscube, n_obs, n_states)

            self.monitor.report(cur_logprob)
            if self.monitor.converged:
                # msg = 'Converged at iteration {} with current logprob {:.2f} and previous logprob {:.2f}'
                # cur_logprob = self.monitor.history[1] if len(self.monitor.history) > 1 else self.monitor.history[0]
                # prev_logprob = self.monitor.history[1] if len(self.monitor.history) > 1 else -1
                # msg = msg.format(it, cur_logprob, prev_logprob)
                # self.logger.debug(msg)
                break

        return self

    def compute_fwd(self):
        raise NotImplementedError('Not yet implemented')

    def compute_bwd(self):
        raise NotImplementedError('Not yet implemented')
