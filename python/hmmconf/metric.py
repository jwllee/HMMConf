import numpy as np
from hmmconf import hmmconf_setup, pm_extra
from hmmconf.utils import make_logger


logger = make_logger('metric.py')


class InjectedDistanceMetric(dict):
    """Observer class for ConformanceStatus that computes the injected distance metric.
    """

    def __init__(self, distmat, callback=None):
        """
        :param distmat: distance matrix
        :param callback: callback function when self.update is called
        """
        self.distmat = distmat
        self.last_mode_state = dict()
        self.last_updated_case = None
        self.time_dict = dict()
        self.callback = callback

    @classmethod
    def create(cls, net, init_marking, is_inv, callback=None):
        rg, inv_states = pm_extra.build_reachability_graph(net, init_marking, is_inv)
        sorted_states = sorted(list(rg.states), key=lambda s: (s.data['disc'], s.name))
        node_map = {key:val for val, key in enumerate(map(lambda state: state.name, sorted_states))}
        int2state = {val:key for key, val in node_map.items()}
        state2int = {val:key for key, val in int2state.items()}

        init = pm_extra.get_init_marking(rg)
        is_inv_rg = lambda t: t.name is None
        distmat = hmmconf_setup.compute_distmat(rg, state2int, is_inv_rg)

        return InjectedDistanceMetric(distmat, callback)

    def update_distance(self, cid, mode_state, dist_dict):
        if cid in dist_dict:
            dist = self.distmat[self.last_mode_state, mode_state]
            dist_dict[cid] = dist_dict[cid] + max(0, dist - 1)
            self.time_dict[cid] += 1
        else:
            dist_dict[cid] = 0
            self.time_dict[cid] = 1

    def update(self, status):
        cid = status.caseid
        event = status.last_event
        mode_state = np.argmax(status.last_logfwd)
        self.update_distance(cid, mode_state, self)
        self.last_mode_state = mode_state
        self.last_updated_case = cid

        if self.callback is not None:
            self.callback(cid, event, self)

    def __repr__(self):
        repr_ = '{}(last_updated: {}, last_dist: {:.3f})'
        last_dist = self[self.last_updated_case] if self.last_updated_case else -1.
        repr_ = repr_.format(InjectedDistanceMetric.__name__, 
                             self.last_updated_case,
                             last_dist)
        return repr_


class CompletenessMetric(InjectedDistanceMetric):
    """Observer class for ConformanceStatus that computes the completeness metric.
    """
    def __init__(self, distmat, callback=None):
        """
        :param distmat: distance matrix
        :param callback: callback function when self.update is called
        """
        super().__init__(distmat, callback)
        self.dist_dict = dict()

    @classmethod
    def create(cls, net, init_marking, is_inv, callback=None):
        rg, inv_states = pm_extra.build_reachability_graph(net, init_marking, is_inv)
        sorted_states = sorted(list(rg.states), key=lambda s: (s.data['disc'], s.name))
        node_map = {key:val for val, key in enumerate(map(lambda state: state.name, sorted_states))}
        int2state = {val:key for key, val in node_map.items()}
        state2int = {val:key for key, val in int2state.items()}

        init = pm_extra.get_init_marking(rg)
        is_inv_rg = lambda t: t.name is None
        distmat = hmmconf_setup.compute_distmat(rg, state2int, is_inv_rg)

        return CompletenessMetric(distmat, callback)

    def update_completeness(self, cid):
        dist = self.dist_dict[cid]
        cid_time = self.time_dict[cid]
        self[cid] = cid_time / (dist + cid_time)
        # info_msg = 'case: {} time: {} dist: {:.2f} completeness: {:.2f}'
        # info_msg = info_msg.format(cid, cid_time, dist, self[cid])
        # logger.info(info_msg)

    def update(self, status):
        cid = status.caseid
        event = status.last_event
        mode_state = np.argmax(status.last_logfwd)
        self.update_distance(cid, mode_state, self.dist_dict)
        self.update_completeness(cid)
        self.last_mode_state = mode_state
        self.last_updated_case = cid

        if self.callback is not None:
            self.callback(cid, event, self)

    def __repr__(self):
        repr_ = '{}(last_updated: {}, last_completeness: {:.3f})'
        last_completeness = self[self.last_updated_case] if self.last_updated_case else -1.
        repr_ = repr_.format(CompletenessMetric.__name__, 
                             self.last_updated_case,
                             last_completeness)
        return repr_
