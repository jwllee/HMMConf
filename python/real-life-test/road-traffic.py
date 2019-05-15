import numpy as np
import pandas as pd
import time, os, argparse, subprocess
import multiprocessing as mp
from pandas.api.types import CategoricalDtype

import hmmconf
from pm4py.objects.petri.importer import pnml as pnml_importer
from pm4py.visualization.transition_system import util


np.set_printoptions(precision=2)


logger = hmmconf.utils.make_logger(__file__)


DATA_DIR = os.path.join('..', '..', 'data', 'road-traffic-fines')
LOG_FP = os.path.join(DATA_DIR, 'Road_Traffic_Fine_Management_Process-v3.xes.gz')
LOG_CSV_FP = os.path.join('.', 'Road_Traffic_Fine_Management_Process-v3.csv')
MODEL_FP = os.path.join(DATA_DIR, 'road-traffic-v2.pnml')
RESULT_DIR = os.path.join('.', 'results')


if not os.path.isdir(RESULT_DIR):
    os.mkdir(RESULT_DIR)


ACTIVITY = 'activity'
ACTIVITY_ID = 'activity_id'
CASEID = 'caseid'


# EM params
N_JOBS = 'n_jobs'
N_ITER = 'n_iter'
TOL = 'tol'
RANDOM_SEED_PARAM = 'random_seed'


HEADER = [
    'fold_id',
    'caseid',
    'activity',
    'activity_id',
    'state_conformance',
    'emission_conformance',
    'final_conformance',
    'sequence_likelihood',
    'most_likely_state',
    'likelihood_mode',
    'inc_mode_distance',
    'sum_mode_distance',
    'mode_completeness',
    'inc_exp_distance',
    'sum_exp_distance',
    'exp_completeness',
    'init_completeness',
    'is_exception'
]


def import_net():
    net, init_marking, final_marking = pnml_importer.import_net(MODEL_FP)
    return net, init_marking, final_marking


def import_log():
    # check if the log has been converted to csv
    if not os.path.exists(LOG_CSV_FP):
        logger.info('Converting XES to CSV...')
        converter_fp = os.path.join('..', '..', 'data', 'xes2csv.py')
        subprocess.run(['python3', converter_fp, '-f', LOG_FP, '-o', LOG_CSV_FP])

    return pd.read_csv(LOG_CSV_FP)


def map_net_activity(net, actmap):
    is_inv = lambda t: t.label is None or t.label.strip() == ''
    for t in net.transitions:
        if not is_inv(t):
            t.label = actmap[t.label]
        else:
            t.label = None


def modify_place_name(net):
    for p in net.places:
        p.name = p.name


def process_net(net, init_marking, final_marking):
    is_inv = lambda t: t.label is None
    inv_trans = list(filter(is_inv, net.transitions))
    logger.info('Number of invisible transitions: {}'.format(len(inv_trans)))
    rg, inv_states = hmmconf.build_reachability_graph(net, init_marking, is_inv)
    is_inv = lambda t: t.name is None
    hmmconf.connect_inv_markings(rg, inv_states, is_inv)
    return rg


def setup_hmm(rg, EM_params):
    G, node_map = hmmconf.rg_to_nx_undirected(rg, map_nodes=True)
    n_states = len(node_map)

    # deal with initial marking and get start probability
    is_inv = lambda t: t.name is None
    startprob = hmmconf.compute_startprob(rg, node_map, n_states, is_inv)
    # add epsilon mass to all states
    startprob += 1e-5
    hmmconf.utils.normalize(startprob, axis=1)

    # remove invisible transitions 
    to_remove = list()
    for t in rg.transitions:
        if is_inv(t):
            to_remove.append(t)

    for t in to_remove:
        rg.transitions.remove(t)
        t.from_state.outgoing.remove(t)
        t.to_state.incoming.remove(t)

    dist_df = hmmconf.compute_distance_matrix(G, node_map, as_dataframe=True)
    distmat = dist_df.values
    # print('Distance df: \n{}'.format(dist_df))

    obsmap = {t.name: int(t.name) for t in rg.transitions}
    int2state = {val:key for key, val in node_map.items()}
    int2obs = {val:key for key, val in obsmap.items()}
    n_obs = len(obsmap)

    logger.info('Obsmap: {}'.format(obsmap))
    logger.info('No. of observations: {}'.format(n_obs))
    logger.info('No. of states: {}'.format(n_states))

    transcube = hmmconf.compute_state_trans_cube(rg, node_map, obsmap, n_obs, n_states)
    emitmat = hmmconf.compute_emission_mat(rg, node_map, obsmap, n_obs, n_states)
    confmat = hmmconf.compute_conformance_mat(emitmat)
    conform_f = hmmconf.conform

    hmm = hmmconf.HMMConf(conform_f, startprob, transcube, emitmat, confmat, distmat, 
                          int2state, int2obs, n_states, n_obs, params='to', verbose=True, 
                          n_jobs=EM_params[N_JOBS], tol=EM_params[TOL], 
                          n_iter=EM_params[N_ITER], random_seed=EM_params[RANDOM_SEED_PARAM])
    return hmm


def make_conformance_tracker(hmm):
    return hmmconf.ConformanceTracker(hmm, max_n_case=100000)


def event_df_to_hmm_format(df):
    lengths = df.groupby(CASEID).count()[ACTIVITY_ID].values
    X = df[[ACTIVITY_ID]].values
    return X, lengths


if __name__ == '__main__':
    start_all = time.time()
    TEST = False
    N_FOLDS = 5
    RANDOM_SEED = 123
    EM_params = {
        N_JOBS: mp.cpu_count() - 1,
        N_ITER: 30,
        TOL: 1,
        RANDOM_SEED_PARAM: RANDOM_SEED
    }

    logger.info('EM params: \n{}'.format(EM_params))

    logger.info('Running real life test...')

    net, init_marking, final_marking = import_net()
    log_df = import_log()

    logger.info('Filter out log activities that are not in net...')
    logger.info('Before filter log df shape: {}'.format(log_df.shape))
    net_actlabels = list(map(lambda t: t.label, net.transitions))
    excl_acts = filter(lambda a: a not in net_actlabels, log_df[ACTIVITY].unique())
    excl_acts = list(excl_acts)
    logger.info('{} event activities not in net: {}'.format(len(excl_acts), excl_acts))
    excl_acts_rows = log_df[ACTIVITY].isin(excl_acts)
    log_df = log_df.loc[~(excl_acts_rows), :]
    # filter out cases that are too short
    TOO_SHORT = 2
    caselen_df = log_df.groupby(CASEID).agg({ACTIVITY: 'count'}).reset_index()
    excl_cids = caselen_df.loc[(caselen_df[ACTIVITY] <= TOO_SHORT), CASEID]
    excl_cids_rows = log_df[CASEID].isin(excl_cids)
    logger.info('{} cases filtered out.'.format(excl_cids.shape[0]))
    log_df = log_df.loc[~(excl_cids_rows), :]

    logger.info('Filtered log df shape: {}'.format(log_df.shape))
    logger.info('Number of cases: {}'.format(log_df[CASEID].unique().shape[0]))

    if len(excl_acts) > 0:
        logger.info('Redo activity_id mapping...')
        ordered_acts = sorted(list(log_df[ACTIVITY].unique()))
        activity_cat_type = CategoricalDtype(categories=ordered_acts, ordered=True)
        log_df[ACTIVITY_ID] = log_df.activity.astype(activity_cat_type).cat.codes

    logger.info('Mapping activity to integer labels...')
    actmap = log_df[[ACTIVITY, ACTIVITY_ID]].set_index(ACTIVITY)
    actmap = actmap.to_dict()[ACTIVITY_ID]
    logger.info(actmap)
    map_net_activity(net, actmap)

    logger.info('Modify place names...')
    modify_place_name(net)

    if TEST:
        caseids = log_df[CASEID].unique()[-100:]
        to_include = log_df[CASEID].isin(caseids)
        n_cases = len(caseids)
        logger.info('Small test on {} cases...'.format(n_cases))
        filtered_df = log_df.loc[to_include,:]
    else:
        filtered_df = log_df

    logger.info('{}-fold cross validation'.format(N_FOLDS))
    caseids = filtered_df[CASEID].unique()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(caseids)
    caseid_kfolds = np.array_split(caseids, N_FOLDS)

    result_rows = list()

    for fold_id, caseid_fold in enumerate(caseid_kfolds):
        train_inds = filtered_df[CASEID].isin(caseid_fold)
        train_event_df = filtered_df.loc[train_inds,:]
        test_event_df = filtered_df.loc[~train_inds,:]

        n_rows_train = train_event_df.shape[0]
        n_rows_test = test_event_df.shape[0]
        total_n_rows = n_rows_train + n_rows_test
        err_msg = 'Sum of no. train ({}) and test df ({}) does not equal total rows ({})'
        err_msg = err_msg.format(n_rows_train, n_rows_test, total_n_rows)
        assert total_n_rows == filtered_df.shape[0], err_msg

        logger.info('EMing on {} cases'.format(len(caseid_fold)))

        logger.info('Process net...')
        rg = process_net(net, init_marking, final_marking)

        logger.info('Setting up HMM...')
        hmm = setup_hmm(rg, EM_params)

        logger.info('Make conformance tracker...')
        tracker = make_conformance_tracker(hmm)

        X, lengths = event_df_to_hmm_format(train_event_df)
        start_fit = time.time()
        tracker.hmm.fit(X, lengths)
        end_fit = time.time()
        took_fit = end_fit - start_fit
        logger.info('Training using {} cases took: {:.2f}s'.format(len(caseid_fold), took_fit))
        start_conf = time.time()
        for row in test_event_df[[CASEID, ACTIVITY, ACTIVITY_ID]].itertuples(index=False):
            caseid = row.caseid
            event = row.activity_id
            act = row.activity

            score = tracker.replay_event(caseid, event)
            conf_arr = score[0]
            ml_state = score[1]
            like_mode = score[2]
            complete = score[3]
            exp_inc_dist = score[4]
            mode_dist = score[5]
            sum_dist = score[6]
            sum_mode_dist = score[7]
            exp_complete = score[8]
            mode_complete = score[9]
            is_exception = score[10]
            seq_likelihood = score[11]

            if TEST:
                msg = '{caseid} replay {event:<3}: ' \
                    'conf: {conf:.2f}, compl: {compl:.2f}, ' \
                    'inc_dist: {inc_dist:.2f}, mode_dist: {mode_dist:.2f}, ' \
                    'sum_dist: {sum_dist:.2f}, sum_mode_dist: {sum_mode_dist:.2f}, ' \
                    'exp_compl: {exp_compl:.2f}, mode_compl: {mode_compl:.2f}, ' \
                    'sequence likelihood: {seq_like:.2f}, {state}, {like:.2f} is_exception: {is_except}'
                msg = msg.format(caseid=caseid, event=act, conf=conf_arr[2],
                                compl=complete, inc_dist=exp_inc_dist,
                                mode_dist=mode_dist, sum_dist=sum_dist,
                                sum_mode_dist=sum_mode_dist, 
                                exp_compl=exp_complete, 
                                mode_compl=mode_complete,
                                state=ml_state, like=like_mode, 
                                is_except=is_exception, seq_like=seq_likelihood)
                logger.info(msg)
            
            result_line = [
                fold_id, 
                caseid,
                act,
                event,
                conf_arr[0],    # state conformance
                conf_arr[1],    # emission conformance
                conf_arr[2],    # final conformance
                seq_likelihood,
                ml_state, 
                like_mode,
                mode_dist,
                sum_mode_dist,
                mode_complete,
                exp_inc_dist,
                sum_dist,
                exp_complete,
                complete,
                is_exception
            ]
            result_rows.append(result_line)

        end_conf = time.time()
        took_conf = end_conf - start_conf
        msg = 'Took {:.2f}s for {} instances, {:.5f}ms per instance.'
        n_test = test_event_df.shape[0]
        msg = msg.format(took_conf, n_test, took_conf / n_test * 1000.)
        logger.info(msg)

    result_fname = 'road-traffic.csv'
    if TEST:
        result_fname = result_fname + '_test'

    result_fp = os.path.join(RESULT_DIR, result_fname)

    result_df = pd.DataFrame.from_records(result_rows, columns=HEADER)
    result_df.to_csv(result_fp, index=None)

    end_all = time.time()
    took_all = end_all - start_all
    logger.info('Test took: {:.2f}s'.format(took_all))
