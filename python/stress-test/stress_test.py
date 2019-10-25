import numpy as np
import pandas as pd
import time, os, argparse, sys
import multiprocessing as mp
from datetime import datetime
from collections import defaultdict

import hmmconf
from hmmconf import metric, pm_extra
from pm4py.objects.petri.importer import pnml as pnml_importer
from pm4py.visualization.transition_system import util


np.set_printoptions(precision=2)


logger = hmmconf.utils.make_logger(__file__)


MODEL_FP = os.path.join('..', '..', 'data', 'BPM2018', 'stress-test', 'model.pnml')
DATA_FP = os.path.join('..', '..', 'data', 'BPM2018', 'stress-test', 'filtered-stream.csv')
RESULT_DIR = os.path.join('results')

ACTIVITY = 'activity'
ACTIVITY_ID = 'activity_id'
CASEID = 'caseid'


if not os.path.isdir(RESULT_DIR):
    os.mkdir(RESULT_DIR)


ACTIVITY = 'activity'
ACTIVITY_ID = 'activity_id'
CASEID = 'caseid'

# EM params
N_JOBS = 'n_jobs'
N_ITER = 'n_iters'
TOL = 'tol'
RANDOM_SEED_PARAM = 'random_seed'
N_FOLDS = 'n_folds'
IS_TEST = 'is_test'
CONF_TOL = 'conformance_tol'
PRIOR_MULTIPLIER = 'prior_multiplier'
EM_PARAMS = 'em_params'
MAX_N_CASE = 'max_n_case'


# experiment configurations
EXPERIMENT_CONFIGS = {
    N_JOBS: mp.cpu_count() - 1,
    N_ITER: 30,
    TOL: 5,
    RANDOM_SEED_PARAM: 123,
    N_FOLDS: 5,
    IS_TEST: False,
    CONF_TOL: 0,
    PRIOR_MULTIPLIER: 1.,
    EM_PARAMS: 'to',
    MAX_N_CASE: 10000
}


def experiment_configs2df(configs):
    items = sorted(list(configs.items()), key=lambda t: t[0])
    columns, values = zip(*items)
    return pd.DataFrame([values], columns=columns)


def get_results_dirname(configs):
    dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    configs_str = ['{}_{}'.format(k, v) for k, v in configs.items()]
    configs_str = '-'.join(configs_str)
    dirname = '{}-{}'.format(dt, configs_str)
    return dirname


def map_net_activity(net, actmap):
    for t in net.transitions:
        if t.label:
            t.label = actmap[t.label]


def modify_place_name(net):
    """Append p to place names so that regular expression can be used to map 
    markings back and forth to its integer representation.
    """
    for p in net.places:
        p.name = p.name


def estimate_conform_params(event_df, state2int, obs2int, 
                            net, init_marking, final_marking,
                            is_inv, add_prior=True, multiplier=1.):
    # group cases
    grouped_by_caseid = event_df.groupby('caseid')
    cases = list()

    for caseid, case_df in grouped_by_caseid:
        case = case_df['activity']
        cases.append((caseid, case))

    results = hmmconf.get_counts_from_log(
        cases, state2int, obs2int,
        net, init_marking, final_marking, is_inv
    )
    trans_count, emit_count, conforming_cid = results

    # get pseudo counts
    if add_prior:
        is_inv_rg = lambda t: t.name is None
        rg, inv_states = hmmconf.build_reachability_graph(net, init_marking, is_inv)
        init = pm_extra.get_init_marking(rg)
        trans_pseudo_count = hmmconf.get_pseudo_counts_transcube(rg, init, 
                                                                 is_inv_rg, 
                                                                 state2int, obs2int, multiplier)
        emit_pseudo_count = hmmconf.get_pseudo_counts_emitmat(rg, init, 
                                                              is_inv_rg, 
                                                              state2int, obs2int, multiplier)

        transcube = hmmconf.estimate_transcube(trans_count, trans_pseudo_count)
        emitmat = hmmconf.estimate_emitmat(emit_count, emit_pseudo_count)
    else:
        transcube = hmmconf.estimate_transcube(trans_count)
        emitmat = hmmconf.estimate_emitmat(emit_count)
        
    return transcube, emitmat, conforming_cid


def event_df_to_hmm_format(df):
    lengths = df.groupby('caseid').count()['activity_id'].values
    X = df[['activity_id']].values
    return X, lengths


def sizeof_status(s):
    size_obj = sys.getsizeof(s)
    size_obj += (s.last_logfwd.size * s.last_logfwd.itemsize)
    return size_obj


def sizeof_hmm(hmm):
    assert isinstance(hmm, hmmconf.HMMConf)
    size_obj = sys.getsizeof(hmm)
    _bytes = 0
    _bytes += hmm.logstartprob.nbytes
    _bytes += hmm.logtranscube.nbytes
    _bytes += hmm.logtranscube_d.nbytes
    _bytes += hmm.logemitmat.nbytes
    _bytes += hmm.logemitmat_d.nbytes
    _bytes += hmm.confmat.nbytes
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


def sizeof_tracker(t):
    assert isinstance(t, hmmconf.ConformanceTracker)
    size_obj = sys.getsizeof(t)
    # avoid double counting
    size_obj -= sys.getsizeof(t.hmm) 
    size_obj += sizeof_hmm(t.hmm)
    # logger.info('Number of cases: {}'.format(len(t.caseid_history)))
    for caseid in t.caseid_history:
        size_obj += sys.getsizeof(caseid)
    for key, status in t.items():
        size_obj += sys.getsizeof(key)
        size_obj += sizeof_status(status)
    return size_obj


def sizeof_tracker_mb(t):
    return sizeof_tracker(t) / 1e6


if __name__ == '__main__':
    print('Start stress test...')
    start = time.time()

    configs_df = experiment_configs2df(EXPERIMENT_CONFIGS)
    info_msg = 'Experiment configuration: \n{}'.format(configs_df)
    logger.info(info_msg)

    results_dirname = get_results_dirname(EXPERIMENT_CONFIGS)
    results_dir = os.path.join(RESULT_DIR, results_dirname)
    os.mkdir(results_dir)

    print('Importing data...')
    net, init_marking, final_marking = pnml_importer.import_net(MODEL_FP)
    net_orig, init_marking_orig, final_marking_orig = pnml_importer.import_net(MODEL_FP)
    log_df = pd.read_csv(DATA_FP)

    print('Mapping activity to integer labels...')
    obs2int = log_df[[ACTIVITY, ACTIVITY_ID]].set_index(ACTIVITY)
    obs2int = obs2int.to_dict()[ACTIVITY_ID]
    int2obs = {key:val for key, val in obs2int.items()}
    obs2int_df = pd.DataFrame(list(obs2int.items()), columns=['activity', 'activity_int'])
    info_msg = 'Activity 2 int dataframe: \n{}'.format(obs2int_df)
    logger.info(info_msg)
    map_net_activity(net, obs2int)

    logger.info('Modify place names...')
    modify_place_name(net)

    if EXPERIMENT_CONFIGS[IS_TEST]:
        caseids = log_df[CASEID].unique()[-100:]
        to_include = log_df[CASEID].isin(caseids)
        n_cases = len(caseids)
        logger.info('Small test on {} cases'.format(n_cases))
        filtered_df = log_df.loc[to_include,:]
    else:
        filtered_df = log_df

    print('Setting up HMM...') 
    is_inv = lambda t: t.label is None
    rg, inv_states = hmmconf.build_reachability_graph(net, init_marking, is_inv)
    sorted_states = sorted(list(rg.states), key=lambda s: (s.data['disc'], s.name))
    node_map = {key:val for val, key in enumerate(map(lambda state: state.name, sorted_states))}
    int2state = {val:key for key, val in node_map.items()}
    state2int = {val:key for key, val in int2state.items()}

    is_inv_rg = lambda t: t.name is None
    init = hmmconf.get_init_marking(rg)
    startprob = hmmconf.compute_startprob(rg, state2int, is_inv_rg)
    conf_obsmap = {i:i for i in obs2int.values()}
    confmat = hmmconf.compute_confmat(rg, init, is_inv_rg, state2int, conf_obsmap)

    params = estimate_conform_params(
        filtered_df, state2int, obs2int, net_orig, init_marking_orig, final_marking_orig, is_inv
    )
    transcube, emitmat, conforming_caseid = params
    hmmconf_params = {
        'params': EXPERIMENT_CONFIGS[EM_PARAMS],
        'conf_tol': EXPERIMENT_CONFIGS[CONF_TOL],
        'n_iter': EXPERIMENT_CONFIGS[N_ITER],
        'tol': EXPERIMENT_CONFIGS[TOL],
        'verbose': True,
        'n_procs': EXPERIMENT_CONFIGS[N_JOBS],
        'random_seed': EXPERIMENT_CONFIGS[RANDOM_SEED_PARAM]
    }
    hmm = hmmconf.HMMConf(startprob, transcube, emitmat, confmat, int2state,
                        int2obs, **hmmconf_params)

    int2state_list = list(int2state.items())
    stateid_list, state_list = zip(*int2state_list)
    columns = ['state_id', 'state']
    state_id_df = pd.DataFrame({
        'state_id': stateid_list,
        'state': state_list
    })
    info_msg = 'State id df: \n{}'.format(state_id_df)
    logger.info(info_msg)

    time_cols = [
        'event', 'n_cases', 'total time', 
        'Average processing time per event',
        'local avg time'
    ]

    mem_cols = [
        'event', 'n_cases', 'Total space used (MB)'
    ]

    # memory test
    print('Doing memory test...')
    print('Make conformance tracker...')
    tracker = hmmconf.ConformanceTracker(hmm, max_n_case=EXPERIMENT_CONFIGS[MAX_N_CASE])

    print('Tracker size: {:.0f}MB'.format(sizeof_tracker_mb(tracker)))

    mem_lines = list()
    mem_lines.append((0, 0, sizeof_tracker_mb(tracker)))

    total_events = 0
    
    for row in filtered_df.itertuples(index=False):
        caseid = row.caseid
        event = row.activity_id
        act = row.activity

        # start_i = time.time()
        result = tracker.replay_event(caseid, event)
        # end_i = time.time()
        # print('Took {:.3f}s'.format(end_i - start_i))

        total_events += 1

        if total_events % 10000 == 0:
            sizetracker = sizeof_tracker_mb(tracker)
            n_cases = len(tracker)
            msg = 'Total events: {}, ' \
                  'Number of cases: {}, ' \
                  'Memory: {:.2f}MB'.format(total_events, n_cases, sizetracker)
            print(msg)
            # start_i = time.time()
            line_i = (total_events, n_cases, sizetracker)
            # end_i = time.time()
            # print('took {:.2f}s to count mem'.format(end_i - start_i))
            mem_lines.append(line_i)

    # time test
    time_lines = list()
    time_lines.append((0, 0, '', '', ''))
    total_events = 0
    total_time = 0
    local_avg = 0

    print('Doing time test...')
    print('Make conformance tracker...')
    tracker = hmmconf.ConformanceTracker(hmm, max_n_case=EXPERIMENT_CONFIGS[MAX_N_CASE])
    for row in filtered_df.itertuples(index=False):
        caseid = row.caseid
        event = row.activity_id
        act = row.activity

        start_i = time.time()
        result = tracker.replay_event(caseid, event)
        end_i = time.time()

        total_time += ((end_i - start_i) * 1000)
        local_avg += ((end_i - start_i) * 1000)
        total_events += 1

        if total_events % 10000 == 0:
            n_cases = len(tracker)
            avg_time = total_time / total_events
            local_avg = local_avg / 10000
            msg = 'Total events: {}, ' \
                  'Number of cases: {}, ' \
                  'Total time: {:.2f}ms, ' \
                  'Average time: {:.2f}ms, ' \
                  'Local average time: {:.2f}ms'
            msg = msg.format(total_events, n_cases, total_time, avg_time, local_avg)
            print(msg)
            line_i = (total_events, n_cases, total_time, avg_time, local_avg)
            time_lines.append(line_i)
            local_avg = 0

    mem_df = pd.DataFrame.from_records(mem_lines, columns=mem_cols)
    time_df = pd.DataFrame.from_records(time_lines, columns=time_cols)

    out_fname = 'results-stress-test.csv'
    out_fp = os.path.join(results_dir, out_fname)
    df = pd.merge(time_df, mem_df, on=['event', 'n_cases'])
    err_msg = 'Merged results dataframe ({}) does not have the same number of rows as mem_df ({}) and time_df ({})'
    err_msg = err_msg.format(df.shape[0], mem_df.shape[0], time_df.shape[0])
    assert df.shape[0] == mem_df.shape[0] and df.shape[0] == time_df.shape[0], err_msg
    df.to_csv(out_fp, index=None)

    end = time.time()
    took = end - start
    print('Stress test took {:.2f}s'.format(took))
