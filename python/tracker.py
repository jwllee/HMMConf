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
    """
    # @todo: add last distance
    def __init__(self, startprob, int2state, int2obs, max_history=1):
        self.logfwd = utils.log_mask_zero(startprob)
        self.state_est = startprob
        self.int2state = int2state
        self.int2obs = int2obs
        self.last_update = None
        self.max_history = max_history
        self.conformance_history = []
        self.activity_history = []

    @property
    def most_likely_state(self):
        """Most likely state according to the current state estimation, i.e., mode of the categorical distribution
        """
        return self.int2state[np.argmax(self.state_est, axis=1)[0]]

    @property
    def last_activity(self):
        """Most recent activity related to the case.
        """
        return self.activity_history[-1] if len(self.activity_history) > 0 else None

    def update(self, act, logfwd, conf_arr):
        """Updates the conformance status of the case.

        :param act int: current activity of the case
        :param logfwd array_like: updated log forward probability following the current activity
        :param conf_arr array_like: conformance array for the current activity
        """
        self.logfwd = logfwd
        self.state_est = logfwd.copy()
        utils.log_normalize(self.state_est, axis=1)
        self.state_est = np.exp(self.state_est)

        self.last_update = dt.now()
        self.activity_history.append(act)
        self.conformance_history.append(conf_arr)

        if len(self.activity_history) > self.max_history:
            self.activity_history.pop(0)
            self.conformance_history.pop(0)


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

    def replay_event(self, caseid, event):
        """Replays event of caseid.

        :param caseid str: caseid 
        :param event int: event
        :return: conformance array, mode of state estimation
        """
        if caseid in self:
            status = self[caseid]
            logfwd, conf_arr = self.hmm.forward(event, status.last_activity, status.logfwd)
            status.update(event, logfwd, conf_arr)
            self.caseid_history.remove(caseid)
        else:
            if (len(self.caseid_history) >= self.max_n_case):
                to_remove = self.caseid_history.popleft()
                del self[to_remove]

            status = ConformanceStatus(self.hmm.startprob, self.hmm.int2state, self.hmm.int2obs)
            self[caseid] = status
            logfwd, conf_arr = self.hmm.forward(event)
            status.update(event, logfwd, conf_arr)

        # current_score = conf_arr[2]
        self.caseid_history.appendleft(caseid)
        return conf_arr, status.most_likely_state
