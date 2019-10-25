import numpy as np
import os
from datetime import datetime as dt
from collections import deque
from scipy.special import logsumexp


from hmmconf.utils import *
from hmmconf.base import *


__all__ = [
    'ConformanceStatus',
    'ConformanceTracker'
]


logger = make_logger('tracker.py')


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
    def __init__(self, caseid):
        self.caseid = caseid
        self.n_events = 0
        self.last_update = None
        self.last_event = None
        self.last_logfwd = None
        self.last_emitconf = None
        self.last_stateconf = None
        self.last_finalconf = None
        self.last_exception = None
        self.observers = list()

    def __repr__(self):
        repr_ = 'ConformanceStatus({caseid}, {n_events}, {logfwd})'
        repr_ = repr_.format(caseid=self.caseid, n_events=self.n_events, logfwd=self.last_logfwd)
        return repr_

    def __str__(self):
        fwd = np.exp(self.last_logfwd)
        str_  = 'Conformance status of {caseid} after {n_events} events:\n'
        str_ += 'last update:    {update}\n'
        str_ += 'last event:     {event}\n'
        str_ += 'last fwd:     \n{fwd}\n'
        str_ += 'last emitconf:  {emitconf}\n'
        str_ += 'last stateconf: {stateconf}\n'
        str_ += 'last finalconf: {finalconf}\n'
        str_ += 'last exception: {exception}\n'
        str_ = str_.format(
            caseid=self.caseid,
            n_events=self.n_events,
            update=self.last_update,
            event=self.last_event,
            fwd=fwd,
            emitconf=self.last_emitconf,
            stateconf=self.last_stateconf,
            finalconf=self.last_finalconf,
            exception=self.last_exception)
        return str_

    #------------------------------------------------------------
    # Observer pattern
    #------------------------------------------------------------
    def register_observer(self, obs):
        self.observers.append(obs)
    
    def remove_observer(self, obs):
        self.observers.remove(obs)

    def notify_observers(self):
        # info_msg = 'Notifying observers of status for case {} on event {}, observers: {}'
        # info_msg = info_msg.format(self.caseid, self.last_event, self.observers)
        # logger.info(info_msg)

        for obs in self.observers:
            obs.update(self)

    def update(self, event, logfwd, emitconf, stateconf, finalconf, exception):
        self.n_events += 1
        self.last_update = dt.now()
        self.last_logfwd = logfwd
        self.last_event = event
        self.last_emitconf = emitconf
        self.last_stateconf = stateconf
        self.last_finalconf = finalconf
        self.last_exception = exception
        self.notify_observers()


class ConformanceTracker(dict):
    """Online conformance tracker for event stream. Follows the API of https://svn.win.tue.nl/trac/prom/browser/Packages/StreamConformance. 
    **Parameters**:
    :param hmm: modified HMM for conformance checking
    :param max_n_case int: maximum number of cases to keep track of

    **Attributes**:
    :param caseid_history: queue that puts caseids with the most recent event first
    :param logger: logger
    """
    def __init__(self, hmm, max_n_case=10000, observers=[]):
        self.hmm = hmm
        self.max_n_case = max_n_case
        self.caseid_history = deque(maxlen=max_n_case)
        self.logger = make_logger(ConformanceTracker.__name__)
        self.observers = observers

    def replay_event(self, caseid, event):
        prev_obs, prev_logfwd = None, None

        if caseid in self:
            status = self[caseid]
            prev_obs = status.last_event
            prev_logfwd = status.last_logfwd
            self.caseid_history.remove(caseid)
        else:
            if len(self.caseid_history) >= self.max_n_case:
                to_remove = self.caseid_history.popleft()
                del self[to_remove]
            status = ConformanceStatus(caseid)

            # register observers of status
            for obs in self.observers:
                status.register_observer(obs)

            self[caseid] = status

        results = self.hmm.compute_logfwd(event, prev_obs, prev_logfwd)
        logfwd = results[0]
        emitconf = results[1]
        stateconf = results[2]
        finalconf = results[3]
        exception = results[4]

        status.update(event, logfwd, emitconf, stateconf, finalconf, exception)
        self.caseid_history.appendleft(caseid)
        return logfwd, finalconf, exception
