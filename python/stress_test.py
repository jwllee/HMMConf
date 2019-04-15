import numpy as np
import pandas as pd
import time, os, sys

import base, lac_setup, pm_extra, tracker, utils
from pm4py.objects.petri.importer import pnml as pnml_importer
import conform as conform_mod

from itertools import chain
from collections import deque


MODEL_FP = os.path.join('..', 'data', 'BPM2018', 'stress-test', 'model.pnml')
DATA_FP = os.path.join('..', 'data', 'BPM2018', 'stress-test', 'filtered-stream.csv')

ACTIVITY = 'activity'
ACTIVITY_ID = 'activity_id'
CASEID = 'caseid'


np.set_printoptions(precision=2)


logger = utils.make_logger(__file__)


def import_data():
    net, init_marking, final_marking = pnml_importer.import_net(MODEL_FP)
    # print('Number of transitions: {}'.format(len(net.transitions)))
    data_df = pd.read_csv(DATA_FP)
    # print(data_df.head())
    return net, init_marking, final_marking, data_df


def map_net_activity(net, actmap):
    for t in net.transitions:
        if t.label:
            t.label = actmap[t.label]


def modify_place_name(net):
    """Append p to place names so that regular expression can be used to map 
    markings back and forth to its integer representation.
    """
    for p in net.places:
        p.name = 'p' + p.name


def process_net(net, init_marking, final_marking):
    is_inv = lambda t: t.label is None
    inv_trans = list(filter(is_inv, net.transitions))
    # print('Number of invisible transitions: {}'.format(len(inv_trans)))
    rg, inv_states = pm_extra.build_reachability_graph(net, init_marking, is_inv)
    is_inv = lambda t: t.name is None
    pm_extra.connect_inv_markings(rg, inv_states, is_inv)
    return rg


def setup_hmm(rg):
    G, node_map = lac_setup.rg_to_nx_undirected(rg, map_nodes=True)
    n_states = len(node_map)

    is_inv = lambda t: t.name is None
    startprob = lac_setup.compute_startprob(rg, node_map, n_states, is_inv)

    # remove invisible transitions connected to initial marking
    init_mark = pm_extra.get_init_marking(rg)
    to_remove = list()
    for t in init_mark.outgoing:
        if is_inv(t):
            to_remove.append(t)
    for t in to_remove:
        rg.transitions.remove(t)
        t.from_state.outgoing.remove(t)
        t.to_state.incoming.remove(t)

    dist_df = lac_setup.compute_distance_matrix(G, node_map, as_dataframe=True)
    distmat = dist_df.values
    # print('Distance df: \n{}'.format(dist_df))

    obsmap = {t.name: int(t.name) for t in rg.transitions}
    int2state = {val:key for key, val in node_map.items()}
    int2obs = {val:key for key, val in obsmap.items()}
    n_obs = len(obsmap)

    logger.info('No. of states: {}'.format(n_states))

    transcube = lac_setup.compute_state_trans_cube(rg, node_map, obsmap, n_obs, n_states)
    emitmat = lac_setup.compute_emission_mat(rg, node_map, obsmap, n_obs, n_states)
    confmat = lac_setup.compute_conformance_mat(emitmat)
    # startprob += 1e-1
    # utils.normalize(startprob, axis=1)
    # startprob = np.zeros((1, n_states)) + 1. / n_states
    # utils.assert_bounded('startprob', np.sum(startprob).ravel()[0], 0., 1.)
    conform_f = conform_mod.conform

    hmm = base.HMMConf(conform_f, startprob, transcube, emitmat, confmat, distmat, 
                       int2state, int2obs, n_states, n_obs, 
                       params='to', verbose=True, n_jobs=7)
    return hmm


def make_conformance_tracker(hmm):
    return tracker.ConformanceTracker(hmm, max_n_case=100)


def event_df_to_hmm_format(df):
    lengths = df.groupby(CASEID).count()[ACTIVITY_ID].values
    X = df[[ACTIVITY_ID]].values
    return X, lengths


def sizeof_hmm(hmm):
    assert isinstance(hmm, base.HMMConf)
    size_obj = sys.getsizeof(hmm)
    _bytes = 0
    _bytes += hmm.startprob.nbytes
    _bytes += hmm.transcube.nbytes
    _bytes += hmm.transcube_d.nbytes
    _bytes += hmm.emitmat.nbytes
    _bytes += hmm.emitmat_d.nbytes
    _bytes += hmm.confmat.nbytes
    _bytes += hmm.distmat.nbytes
    # print('numpy array size: {}MB'.format(_bytes))
    size_obj += _bytes

    size_int2obs = 0
    for key, val in hmm.int2obs.items():
        size_int2obs += sys.getsizeof(key)
        size_int2obs += sys.getsizeof(val)

    size_int2state = 0
    for key, val in hmm.int2state.items():
        size_int2state += sys.getsizeof(key)
        size_int2state += sys.getsizeof(val)

    size_obj += size_int2obs
    size_obj += size_int2state

    # print('int2obs size: {}MB'.format(size_int2obs))
    # print('int2state size: {}MB'.format(size_int2state))

    return size_obj

# Follows https://docs.python.org/3/library/sys.html
def sizeof_tracker(t):
    assert isinstance(t, tracker.ConformanceTracker)
    size_obj = sys.getsizeof(t) 
    # avoid double counting
    size_obj -= sys.getsizeof(t.hmm) 
    size_obj += sizeof_hmm(t.hmm)
    for caseid in t.caseid_history:
        size_obj += sys.getsizeof(caseid)
    for key, item in t.items():
        size_obj += sys.getsizeof(key)
        size_obj += sys.getsizeof(item)
    return size_obj


def sizeof_tracker_mb(t):
    return sizeof_tracker(t) / 1e6


if __name__ == '__main__':
    print('Start stress test...')
    start = time.time()

    print('Importing data...')
    net, init_marking, final_marking, data_df = import_data()

    print('Mapping activity to integer labels...')
    actmap = data_df[[ACTIVITY, ACTIVITY_ID]].set_index(ACTIVITY).to_dict()[ACTIVITY_ID]
    rev_actmap = {key:val for key, val in actmap.items()}
    map_net_activity(net, actmap)

    print('Modify place names...')
    modify_place_name(net)

    print('Process net...')
    rg = process_net(net, init_marking, final_marking)

    print('Setting up HMM...')
    hmm = setup_hmm(rg)

    print('Make conformance tracker...')
    _tracker = make_conformance_tracker(hmm)

    caseids = data_df[CASEID].unique()[-100:]
    to_include = data_df[CASEID].isin(caseids)
    # caseids = ['case_34'] # warm start example
    # caseids = data_df[CASEID].unique()[:100]
    # n_cases = len(caseids)
    # em_to_include = data_df[CASEID].isin(caseids)

    filtered_df = data_df
    # em_filtered_df = data_df.loc[em_to_include,:]

    print('data df shape: {}'.format(filtered_df.shape))

    # EM training
    # print('EMing...')
    # X, lengths = event_df_to_hmm_format(em_filtered_df)
    # fit_start = time.time()
    # tracker.hmm.fit(X, lengths)
    # fit_end = time.time()
    # fit_took = fit_end - fit_start
    # logger.info('Training {} cases took: {:.2f}s'.format(n_cases, fit_took))

    time_cols = [
        'event', 'total time', 
        'Average processing time per event',
        'local avg time'
    ]

    mem_cols = [
        'event', 'Total space used (MB)'
    ]

    print('Tracker size: {:.0f}MB'.format(sizeof_tracker_mb(_tracker)))

    mem_lines = list()
    mem_lines.append((0, int(sizeof_tracker_mb(_tracker))))

    total_events = 0
    
    # memory test
    print('Doing memory test...')
    for row in filtered_df.itertuples(index=False):
        caseid = row.caseid
        event = row.activity_id
        act = row.activity

        # start_i = time.time()
        score = _tracker.replay_event(caseid, event)
        # end_i = time.time()
        # print('Took {:.3f}s'.format(end_i - start_i))

        total_events += 1

        if total_events % 10000 == 0:
            print('Total events: {}'.format(total_events))
            # start_i = time.time()
            line_i = (total_events, int(sizeof_tracker_mb(_tracker)))
            # end_i = time.time()
            # print('took {:.2f}s to count mem'.format(end_i - start_i))
            mem_lines.append(line_i)

    # time test
    time_lines = list()
    time_lines.append((0, '', '', ''))
    total_events = 0
    total_time = 0
    local_avg = 0

    print('Doing time test...')
    for row in filtered_df.itertuples(index=False):
        caseid = row.caseid
        event = row.activity_id
        act = row.activity

        start_i = time.time()
        score = _tracker.replay_event(caseid, event)
        end_i = time.time()

        total_time += ((end_i - start_i) * 1000)
        local_avg += ((end_i - start_i) * 1000)
        total_events += 1

        if total_events % 10000 == 0:
            print('Total events: {}'.format(total_events))
            avg_time = total_time / total_events
            local_avg = local_avg / 10000
            line_i = (total_events, total_time, avg_time, local_avg)
            time_lines.append(line_i)
            local_avg = 0

    mem_df = pd.DataFrame.from_records(mem_lines, columns=mem_cols)
    time_df = pd.DataFrame.from_records(time_lines, columns=time_cols)

    out_fp = 'results-stress-test.csv'
    df = pd.merge(time_df, mem_df, on='event')
    df.to_csv(out_fp, index=None)

    end = time.time()
    took = end - start
    print('Stress test took {:.2f}s'.format(took))
