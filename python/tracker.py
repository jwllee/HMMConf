import numpy as np
import os
from datetime import datetime as dt
from collections import deque

import base, utils


class ConformanceStatus:
    """Conformance status for a particular case. Follows the API of https://svn.win.tue.nl/trac/prom/browser/Packages/StreamConformance. Keeps track of different interesting conformance detail for a case.

    **Parameters**:
    :param startprob array_like: initial state estimation for the case
    :param int2state dict: mapping from state index to state name
    :param int2obs dict: mapping from observation index to observation name
    :param max_history int: maximum number of historical data to keep where unit is by event

    **Attributes**:
    :param state_est array_like: latest state estimation 
    :param logfwd array_like: latest log foward probability
    :param last_update datetime: last time that the case received an event, ``None`` at the initialization
    :param conformance_history list: list of previous conformance results
    :param activity_history list: list of previous activity for the case
    :param n_events int: number of events in case
    """
    # @todo: add last distance
    def __init__(self, startprob, int2state, int2obs, max_history=1):
        self.startprob = startprob
        self.logfwd = utils.log_mask_zero(startprob)
        self.state_est = startprob
        self.int2state = int2state
        self.int2obs = int2obs
        self.last_update = None
        self.max_history = max_history
        self.completeness_history = deque(maxlen=max_history)
        self.conformance_history = deque(maxlen=max_history)
        self.activity_history = deque(maxlen=max_history)
        self.inc_dist_history = deque(maxlen=max_history)
        self.mode_dist_history = deque(maxlen=max_history)
        self.state_est_history = deque(maxlen=max_history)
        self.sum_dist = 0.
        self.sum_mode_dist = 0.
        self.n_events = 0

    @property
    def most_likely_state(self):
        """Most likely state according to the current state estimation, i.e., mode of the categorical distribution
        """
        return self.int2state[np.argmax(self.state_est, axis=1)[0]]

    @property
    def likelihood_mode(self):
        return np.max(self.state_est)

    @property
    def last_activity(self):
        """Most recent activity related to the case.
        """
        return self.activity_history[-1] if len(self.activity_history) > 0 else None

    def update(self, act, logfwd, conf_arr, complete, inc_dist, mode_inc_dist):
        """Updates the conformance status of the case.

        :param act int: current activity of the case
        :param logfwd array_like: updated log forward probability following the current activity
        :param conf_arr array_like: conformance array for the current activity
        :param complete float: completenss compared with the initial marking
        :param inc_dist float: expected incremental distance
        :param mode_inc_dist float: incremental distance between most likely states from time t and t - 1
        """
        self.n_events += 1
        self.logfwd = logfwd.copy()
        self.state_est = logfwd.copy()
        utils.exp_log_normalize(self.state_est, axis=1)

        self.last_update = dt.now()
        self.state_est_history.append(self.state_est)
        self.activity_history.append(act)
        self.conformance_history.append(conf_arr)
        self.completeness_history.append(complete)
        self.inc_dist_history.append(inc_dist)
        self.mode_dist_history.append(mode_inc_dist)
        self.sum_dist += max(0, inc_dist - 1)
        self.sum_mode_dist += max(0, mode_inc_dist - 1)


class ConformanceTracker(dict):
    """Online conformance tracker for event stream. Follows the API of https://svn.win.tue.nl/trac/prom/browser/Packages/StreamConformance. 

    **Parameters**:
    :param hmm: modified HMM for conformance checking
    :param max_n_case int: maximum number of cases to keep track of

    **Attributes**:
    :param caseid_history: queue that puts caseids with the most recent event first
    :param logger: logger
    """

    def __init__(self, hmm, max_n_case=10000):
        self.hmm = hmm
        self.max_n_case = max_n_case
        self.caseid_history = deque(maxlen=max_n_case)
        self.logger = utils.make_logger(self.__class__.__name__)

    def __compute_completeness(self, n_events, logfwd):
        initstate = self.hmm.startprob
        dist_from_initstate = self.hmm.compute_distance_from_initstate(initstate, logfwd)
        complete = (n_events + 1) / (dist_from_initstate + 1)
        complete = min(1., complete)
        return complete

    def __compute_mode_dist(self, logfwd, prev_logfwd):
        most_likely_state = np.argmax(logfwd)
        prev_most_likely_state = np.argmax(self.hmm.startprob) if prev_logfwd is None else np.argmax(prev_logfwd)
        mode_dist = self.hmm.distmat[prev_most_likely_state,most_likely_state]
        return mode_dist

    def replay_event(self, caseid, event):
        """Replays event of caseid.

        :param caseid str: caseid 
        :param event int: event
        :return: conformance array, mode of state estimation
        """
        if caseid in self:
            status = self[caseid]
            prev_obs = status.last_activity
            prev_logfwd = status.logfwd
            self.caseid_history.remove(caseid)
        else:
            if (len(self.caseid_history) >= self.max_n_case):
                to_remove = self.caseid_history.popleft()
                del self[to_remove]
            status = ConformanceStatus(self.hmm.startprob, 
                                       self.hmm.int2state, 
                                       self.hmm.int2obs)
            self[caseid] = status
            prev_obs, prev_logfwd = None, None

        logfwd, conf_arr = self.hmm.forward(event, prev_obs, prev_logfwd)
        exp_inc_dist = self.hmm.compute_expected_inc_distance(event, logfwd,
                                                              prev_obs, prev_logfwd)
        mode_dist = self.__compute_mode_dist(logfwd, prev_logfwd)
        complete = self.__compute_completeness(status.n_events, logfwd)

        status.update(event, logfwd, conf_arr, complete, exp_inc_dist, mode_dist)
        self.caseid_history.appendleft(caseid)

        score = (conf_arr, status.most_likely_state,
                 status.likelihood_mode, complete, 
                 exp_inc_dist, mode_dist, 
                 status.sum_dist, status.sum_mode_dist)
        return score
