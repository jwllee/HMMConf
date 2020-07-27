import numpy as np
import pandas as pd
import time, os, argparse
import multiprocessing as mp
from datetime import datetime
from collections import defaultdict

import hmmconf
from hmmconf import metric, pm_extra
from pm4py.objects.petri.importer import pnml as pnml_importer
from pm4py.visualization.transition_system import util

logger = hmmconf.utils.make_logger(__file__)
np.set_printoptions(precision=2)


MODEL_FP = os.path.join('..', '..', 'data', 'BPM2018', 'stress-test', 'model.pnml')
DATA_FP = os.path.join('..', '..', 'data', 'BPM2018', 'stress-test', 'filtered-stream.csv')
RESULT_DIR = os.path.join('results', 'time-offline-training')


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
ADD_PRIOR = 'add_prior'

# time results
TIME_IMPORT_DATA = "import_data_secs"
TIME_BUILD_RG = "took_rg_secs"
TIME_CONFORM = "conform_secs"
TIME_FIT = "fit_secs"
TIME_N_TRAIN_EVENTS = "n_train_events"
TIME_N_TRAIN_CASES = "n_train_cases"
TIME_ALL = "all_secs"
TIME_N_EVENTS = "n_events"
TIME_N_CASES = "n_cases"

# experiment configurations
EXPERIMENT_CONFIGS = {
    N_JOBS: mp.cpu_count() - 1,
    N_ITER: 10,
    TOL: 5,
    RANDOM_SEED_PARAM: 123,
    IS_TEST: False,
    CONF_TOL: 0,
    ADD_PRIOR: True,
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


class ConformanceObserver:
    def __init__(self):
        self.emitconf = defaultdict(list)
        self.stateconf = defaultdict(list)

    def update(self, status):
        self.emitconf[status.caseid].append(status.last_emitconf)
        self.stateconf[status.caseid].append(status.last_stateconf)


if __name__ == '__main__':
    logger.info("Start script...")
    start_all = time.time()

    configs_df = experiment_configs2df(EXPERIMENT_CONFIGS)
    logger.info(f"Experiment configuration: \n{configs_df}")

    results_dirname = get_results_dirname(EXPERIMENT_CONFIGS)
    results_dir = os.path.join(RESULT_DIR, results_dirname)
    os.makedirs(results_dir)

    time_dict = dict()

    logger.info("Importing data...")
    start_import = time.time()
    net, init_marking, final_marking = pnml_importer.import_net(MODEL_FP)
    net_orig, init_marking_orig, final_marking_orig = pnml_importer.import_net(MODEL_FP)
    event_df = pd.read_csv(DATA_FP)
    took_import = time.time() - start_import
    time_dict[TIME_IMPORT_DATA] = took_import
    logger.info(f"Importing data took: {took_import:.3f}s")

    logger.info(f"Event df shape: {event_df.shape}")

    logger.info('Mapping activity to integer labels...')
    obs2int = event_df[[ACTIVITY, ACTIVITY_ID]].set_index(ACTIVITY)
    obs2int = obs2int.to_dict()[ACTIVITY_ID]
    int2obs = {key:val for key, val in obs2int.items()}
    obs2int_df = pd.DataFrame(list(obs2int.items()), columns=['activity', 'activity_int'])
    logger.info(f"Activity 2 int dataframe: \n{obs2int_df}")
    map_net_activity(net, obs2int)

    if EXPERIMENT_CONFIGS[IS_TEST]:
        caseids = event_df[CASEID].unique()[-10:]
        to_include = event_df[CASEID].isin(caseids)
        n_cases = len(caseids)
        logger.info(f"Small test on {n_cases} cases")
        filtered_event_df = event_df.loc[to_include,:]
    else:
        filtered_event_df = event_df

    logger.info("Setting up HMM...")
    start_rg = time.time()
    is_inv = lambda t: t.label is None
    rg, inv_states = hmmconf.build_reachability_graph(net, init_marking, is_inv)
    sorted_states = sorted(list(rg.states), key=lambda s: (s.data['disc'], s.name))
    node_map = {key:val for val, key in enumerate(map(lambda state: state.name, sorted_states))}
    int2state = {val:key for key, val in node_map.items()}
    state2int = {val:key for key, val in int2state.items()}
    took_rg = time.time() - start_rg
    time_dict[TIME_BUILD_RG] = took_rg
    logger.info(f"Building reachability graph took: {took_rg:.3f}s")

    is_inv_rg = lambda t: t.name is None
    init = hmmconf.get_init_marking(rg)
    startprob = hmmconf.compute_startprob(rg, state2int, is_inv_rg)
    conf_obsmap = {i:i for i in obs2int.values()}
    confmat = hmmconf.compute_confmat(rg, init, is_inv_rg, state2int, conf_obsmap)

    logger.info("Estimating conformining parameters")
    start_conform = time.time()
    params = estimate_conform_params(
        filtered_event_df,
        state2int, obs2int,
        net_orig, init_marking_orig, final_marking_orig, is_inv,
        add_prior=EXPERIMENT_CONFIGS[ADD_PRIOR],
        multiplier=EXPERIMENT_CONFIGS[PRIOR_MULTIPLIER],
    )
    took_conform = time.time() - start_conform
    time_dict[TIME_CONFORM] = took_conform
    time_dict[TIME_N_CASES] = filtered_event_df["caseid"].unique().shape[0]
    time_dict[TIME_N_EVENTS] = filtered_event_df.shape[0]
    logger.info(f"Estimating conforming distribution params took: {took_conform:.3f}s")
    transcube, emitmat, conforming_caseids = params
    hmmconf_params = {
        'params': EXPERIMENT_CONFIGS[EM_PARAMS],
        'conf_tol': EXPERIMENT_CONFIGS[CONF_TOL],
        'n_iter': EXPERIMENT_CONFIGS[N_ITER],
        'tol': EXPERIMENT_CONFIGS[TOL],
        'verbose': True,
        'n_procs': EXPERIMENT_CONFIGS[N_JOBS],
        'random_seed': EXPERIMENT_CONFIGS[RANDOM_SEED_PARAM]
    }
    hmm = hmmconf.HMMConf(
        startprob, transcube, emitmat, confmat, int2state, int2obs, **hmmconf_params
    )

    int2state_list = list(int2state.items())
    stateid_list, state_list = zip(*int2state_list)
    columns = ['state_id', 'state']
    state_id_df = pd.DataFrame({
        'state_id': stateid_list,
        'state': state_list
    })
    logger.info(f"State id df: \n{state_id_df}")

    n_caseids = filtered_event_df["caseid"].unique().shape[0]
    filter_by_conforming_caseids = filtered_event_df["caseid"].isin(conforming_caseids)
    filtered_train_event_df = filtered_event_df.loc[
        ~filter_by_conforming_caseids, :
    ]
    n_caseids_train = filtered_train_event_df["caseid"].unique().shape[0]
    logger.info(f"Filtered train event df shape: {filtered_train_event_df.shape}")
    logger.info(f'Fitting with {n_caseids_train}/{n_caseids} non-conforming cases')

    train_X, train_lengths = event_df_to_hmm_format(filtered_train_event_df)

    logger.info(f"Starting training...")
    start_fit = time.time()
    tracker = hmmconf.ConformanceTracker(hmm, max_n_case=EXPERIMENT_CONFIGS[MAX_N_CASE])
    tracker.hmm.fit(train_X, train_lengths)
    took_fit = time.time() - start_fit
    time_dict[TIME_FIT] = took_fit
    time_dict[TIME_N_TRAIN_CASES] = n_caseids_train
    time_dict[TIME_N_TRAIN_EVENTS] = filtered_train_event_df.shape[0]
    info_msg = f"Training using {n_caseids_train} cases took: {took_fit:.3f}s"
    info_msg += f" ({took_fit / 60:.0f} mins {took_fit % 60:.0f} secs)"
    logger.info(info_msg)

    took_all = time.time() - start_all
    logger.info(f"Took: {took_all / 60:.0f} mins {took_all % 60:.0f} secs")
    time_dict[TIME_ALL] = took_all

    time_fp = os.path.join(results_dir, "time_results.csv") 
    time_df = pd.DataFrame(time_dict, index=[0])
    time_df.to_csv(time_fp, float_format="%.5f", index=None)
