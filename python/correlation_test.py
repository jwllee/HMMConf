import numpy as np
import pandas as pd
import time, os, argparse
import multiprocessing as mp

import hmmconf
from pm4py.objects.petri.importer import pnml as pnml_importer
from pm4py.visualization.transition_system import util


np.set_printoptions(precision=2)


logger = hmmconf.utils.make_logger(__file__)


MODEL_DIR = os.path.join('..', 'data', 'BPM2018', 'correlation-tests', 'models')
LOG_DIR = os.path.join('..', 'data', 'BPM2018', 'correlation-tests', 'logs', 'processed')
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


def choose_net_by_id(idstr):
    matches = list()
    suffix = '{}.pnml'.format(idstr)
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(suffix):
            matches.append(fname)
    assert len(matches) == 1, 'Matches: {}'.format(matches)
    return matches[0]


def choose_log_by_noise(netname, trace_noise_perc, event_noise_perc):
    matches = list()
    prefix = 'log_{}'.format(netname.replace('_reduced', ''))
    suffix = 'noise_trace_{}_noise_event_{}.csv'.format(trace_noise_perc, event_noise_perc)
    for fname in os.listdir(LOG_DIR):
        if fname.startswith(prefix) and fname.endswith(suffix):
            matches.append(fname)
    assert len(matches) == 1, 'Matches: {}'.format(matches)
    return matches[0]


def import_data(net_fname, log_fname):
    net_fp = os.path.join(MODEL_DIR, net_fname)
    log_fp = os.path.join(LOG_DIR, log_fname)

    data_df = pd.read_csv(log_fp)
    net, init_marking, final_marking = pnml_importer.import_net(net_fp)
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
        p.name = p.name


def process_net(net, init_marking, final_marking):
    is_inv = lambda t: t.label is None
    inv_trans = list(filter(is_inv, net.transitions))
    print('Number of invisible transitions: {}'.format(len(inv_trans)))
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
    return hmmconf.ConformanceTracker(hmm)


def event_df_to_hmm_format(df):
    lengths = df.groupby(CASEID).count()[ACTIVITY_ID].values
    X = df[[ACTIVITY_ID]].values
    return X, lengths


def get_log_names(net_fname):
    log_fnames = []
    for trace_noise in range(1, 6):
        for event_noise in range(1, 6):
            log_fname = 'log_{net}_noise_trace_0.{trace}_noise_event_0.{event}.csv'
            log_fname = log_fname.format(net=net_fname,
                                         trace=trace_noise,
                                         event=event_noise)
            log_fnames.append(log_fname)
    return log_fnames


logger = hmmconf.utils.make_logger(__file__)


if __name__ == '__main__':
    start_all = time.time()
    TEST = False
    N_FOLDS = 5
    RANDOM_SEED = 123
    EM_params = {
        N_JOBS: mp.cpu_count(),
        N_ITER: 10,
        TOL: 5,
        RANDOM_SEED_PARAM: RANDOM_SEED
    }

    logger.info('EM params: \n{}'.format(EM_params))

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', action='store',
                        dest='netfiles',
                        help='List of net files to run')

    args = parser.parse_args()

    if args.netfiles is None:
        logger.info('Run as python ./correlation_test.py -f [netfiles]')
        exit(0)

    with open(args.netfiles, 'r') as f:
        netfiles = f.readlines()
    netfiles = list(map(lambda name: name.strip(), netfiles))

    logger.info('Running correlation test on: \n{}'.format(netfiles))

    for net_fname in netfiles:
        start_net = time.time()
        # loop over possible noise combinations
        for log_fname in get_log_names(net_fname):
            start_log = time.time()
            logger.info('Starting test on {}'.format(log_fname))

            logger.info('Importing data...')
            net, init_marking, final_marking, data_df = import_data(net_fname, log_fname)

            logger.info('Mapping activity to integer labels...')
            actmap = data_df[[ACTIVITY, ACTIVITY_ID]].set_index(ACTIVITY)
            actmap = actmap.to_dict()[ACTIVITY_ID]
            rev_actmap = {key:val for key, val in actmap.items()}
            map_net_activity(net, actmap)

            logger.info('Modify place names...')
            modify_place_name(net)

            logger.info('Process net...')
            rg = process_net(net, init_marking, final_marking)

            logger.info('Setting up HMM...')
            hmm = setup_hmm(rg, EM_params)

            logger.info('Make conformance tracker...')
            tracker = make_conformance_tracker(hmm)

            if TEST:
                caseids = data_df[CASEID].unique()[-100:]
                to_include = data_df[CASEID].isin(caseids)
                n_cases = len(caseids)
                logger.info('Small test on {} cases...'.format(n_cases))

                filtered_df = data_df.loc[to_include,:]
            else:
                filtered_df = data_df

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

                X, lengths = event_df_to_hmm_format(train_event_df)
                start_fit = time.time()
                tracker.hmm.fit(X, lengths)
                end_fit = time.time()
                took_fit = end_fit - start_fit
                logger.info('Training using {} cases took: {:.2f}s'.format(len(caseid_fold), took_fit))
                for row in test_event_df.itertuples(index=False):
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

                    if TEST:
                        msg = '{caseid} replay {event:<3}: ' \
                            'conf: {conf:.2f}, compl: {compl:.2f}, ' \
                            'inc_dist: {inc_dist:.2f}, mode_dist: {mode_dist:.2f}, ' \
                            'sum_dist: {sum_dist:.2f}, sum_mode_dist: {sum_mode_dist:.2f}, ' \
                            'exp_compl: {exp_compl:.2f}, mode_compl: {mode_compl:.2f}, ' \
                            '{state}, {like:.2f} is_exception: {is_except}'
                        msg = msg.format(caseid=caseid, event=act, conf=conf_arr[2],
                                        compl=complete, inc_dist=exp_inc_dist,
                                        mode_dist=mode_dist, sum_dist=sum_dist,
                                        sum_mode_dist=sum_mode_dist, 
                                        exp_compl=exp_complete, 
                                        mode_compl=mode_complete,
                                        state=ml_state, like=like_mode, 
                                        is_except=is_exception)
                        logger.info(msg)
                    
                    result_line = [
                        fold_id, 
                        caseid,
                        act,
                        event,
                        conf_arr[0],    # state conformance
                        conf_arr[1],    # emission conformance
                        conf_arr[2],    # final conformance
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

            if TEST:
                log_fname = log_fname + '_test'
            result_fp = os.path.join(RESULT_DIR, log_fname)

            result_df = pd.DataFrame.from_records(result_rows, columns=HEADER)
            result_df.to_csv(result_fp, index=None)

            end_log = time.time()
            took_log = end_log - start_log
            logger.info('Took {:.2f}s for {}'.format(took_log, log_fname))

        end_net = time.time()
        took_net = end_net - start_net
        logger.info('Took {:.2f}s for {}'.format(took_net, net_fname))

    end_all = time.time()
    took_all = end_all - start_all
    logger.info('Test took: {:.2f}s'.format(took_all))
