import numpy as np
import os
from datetime import datetime as dt
from collections import deque

import base, utils


class ConformanceStatus:
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
        return self.int2state[np.argmax(self.state_est, axis=1)[0]]

    @property
    def last_activity(self):
        return self.activity_history[-1] if len(self.activity_history) > 0 else None

    def update(self, act, logfwd, conf_arr):
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
    def __init__(self, hmm, max_n_case=10000):
        self.hmm = hmm
        self.max_n_case = max_n_case
        self.caseid_history = deque(maxlen=max_n_case)
        self.logger = utils.make_logger(self.__class__.__name__)

    def replay_event(self, caseid, event):
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
