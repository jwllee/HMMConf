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


np.set_printoptions(precision=2)


logger = hmmconf.utils.make_logger(__file__)


MODEL_DIR = os.path.join('..', '..', 'data', 'BPM2018', 'correlation-tests', 'models')
LOG_DIR = os.path.join('..', '..', 'data', 'BPM2018', 'correlation-tests', 'logs', 'processed')
RESULT_DIR = os.path.join('..', 'results')


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


def import_net(net_fname):
    net_fp = os.path.join(MODEL_DIR, net_fname)
    net, init_marking, final_marking = pnml_importer.import_net(net_fp)
    return net, init_marking, final_marking


def import_log(log_fname):
    log_fp = os.path.join(LOG_DIR, log_fname)
    return pd.read_csv(log_fp)


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
    start_all = time.time()
    configs_df = experiment_configs2df(EXPERIMENT_CONFIGS)
    info_msg = 'Experiment configuration: \n{}'.format(configs_df)
    logger.info(info_msg)

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store',
                        dest='netfiles',
                        help='List of net files to run')
    args = parser.parse_args()

    if args.netfiles is None:
        err_msg = 'Run as python ./correlation_test.py -f [netfiles]'
        logger.info(err_msg)
        exit(0)

    results_dirname = get_results_dirname(EXPERIMENT_CONFIGS)
    results_dir = os.path.join(RESULT_DIR, results_dirname)
    os.mkdir(results_dir)

    with open(args.netfiles, 'r') as f:
        netfiles = f.readlines()
    netfiles = list(map(lambda name: name.strip(), netfiles))

    info_msg = 'Running correlation test on: \n{}'.format(netfiles)
    logger.info(info_msg)

    for net_fname in netfiles:
        start_net = time.time()
        # create a separate result directory for net_fname
        results_dir_net = os.path.join(results_dir, net_fname.replace('.pnml', ''))
        os.mkdir(results_dir_net)

        net, init_marking, final_marking = import_net(net_fname)
        # unmapped transition labels
        net_orig, init_marking_orig, final_marking_orig = import_net(net_fname)

        log_df_list = list()

        for log_fname in get_log_names(net_fname):
            log_df = import_log(log_fname)
            log_df['log_name'] = log_fname
            log_df['_caseid'] = log_df[CASEID]
            log_df[CASEID] = log_df['log_name'].str.cat(log_df[CASEID].astype(str), sep=':')
            log_df_list.append(log_df)

        # combining all the event logs into one giant event log
        log_df = pd.concat(log_df_list)
        log_df = log_df.reset_index(drop=True)
        case_prefix_df = log_df.copy()
        case_prefix_df = log_df.groupby(CASEID)[ACTIVITY].apply(lambda x: (x + ';').cumsum().str.strip(';'))
        case_prefix_df = case_prefix_df.reset_index(drop=True).to_frame()
        case_prefix_df.columns = ['case_prefix']
        info_msg = 'Case prefix df: \n{}'.format(case_prefix_df.head())
        logger.info(info_msg)
        log_df = pd.concat([log_df, case_prefix_df], axis=1)
        info_msg = 'Event df: \n{}'.format(log_df.head())
        logger.info(info_msg)

        logger.info('Mapping activity to integer labels...')
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
            logger.info('Small test on {} cases...'.format(n_cases))

            filtered_df = log_df.loc[to_include,:]
        else:
            filtered_df = log_df

        n_folds = EXPERIMENT_CONFIGS[N_FOLDS]
        logger.info('{}-fold cross validation'.format(n_folds))
        caseids = filtered_df[CASEID].unique()
        np.random.seed(EXPERIMENT_CONFIGS[RANDOM_SEED_PARAM])
        np.random.shuffle(caseids)
        caseid_kfolds = np.array_split(caseids, n_folds)

        # save things to store
        store_fp = os.path.join(results_dir_net, 'results_store.h5')
        store = pd.HDFStore(store_fp, mode='w')
        store['config_df'] = configs_df
        store['activityid_df'] = obs2int_df
        store['case_prefix_df'] = case_prefix_df

        result_rows = list()

        for fold_id, caseid_fold in enumerate(caseid_kfolds):
            test_inds = filtered_df[CASEID].isin(caseid_fold)
            train_event_df = filtered_df.loc[~test_inds,:]
            test_event_df = filtered_df.loc[test_inds,:]

            n_rows_train = train_event_df.shape[0]
            n_rows_test = test_event_df.shape[0]
            total_n_rows = n_rows_train + n_rows_test
            err_msg = 'Sum of no. train ({}) and test df ({}) does not equal total rows ({})'
            err_msg = err_msg.format(n_rows_train, n_rows_test, total_n_rows)
            assert total_n_rows == filtered_df.shape[0], err_msg

            logger.info('EMing on {} cases'.format(len(caseid_fold)))
            is_inv = lambda t: t.label is None
            rg, inv_states = hmmconf.build_reachability_graph(net, init_marking, is_inv)
            sorted_states = sorted(list(rg.states), key=lambda s: (s.data['disc'], s.name))
            node_map = {key:val for val, key in enumerate(map(lambda state: state.name, sorted_states))}
            int2state = {val:key for key, val in node_map.items()}
            state2int = {val:key for key, val in int2state.items()}

            logger.info('Setting up HMM...')
            is_inv_rg = lambda t: t.name is None
            init = hmmconf.get_init_marking(rg)
            startprob = hmmconf.compute_startprob(rg, state2int, is_inv_rg)
            conf_obsmap = {i:i for i in obs2int.values()}
            confmat = hmmconf.compute_confmat(rg, init, is_inv_rg, state2int, conf_obsmap)

            params = estimate_conform_params(
                train_event_df, state2int, obs2int, net_orig, init_marking_orig, final_marking_orig, is_inv
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

            logger.info('Make conformance tracker...')
            # add metrics as observers
            injected_dist_rows = list()
            def injected_distance_callback(caseid, event, metric):
                case_prefix = event
                if injected_dist_rows:
                    last_row = injected_dist_rows[-1]
                    if last_row[0] == caseid:
                        case_prefix = ', '.join([str(last_row[1]), str(event)])
                injected_dist_rows.append((caseid, case_prefix, metric[caseid]))

            injected_distance = metric.InjectedDistanceMetric.create(net, init_marking, is_inv, 
                                                                     injected_distance_callback)
            completeness_rows = list()
            def completeness_callback(caseid, event, metric):
                case_prefix = event
                if injected_dist_rows:
                    last_row = injected_dist_rows[-1]
                    if last_row[0] == caseid:
                        case_prefix = ', '.join([str(last_row[1]), str(event)])
                completeness_rows.append((caseid, case_prefix, metric[caseid]))

            completeness = metric.CompletenessMetric.create(net, init_marking, is_inv,
                                                            completeness_callback)
            conf_observer = ConformanceObserver()
            observers = [conf_observer, injected_distance, completeness]

            tracker = hmmconf.ConformanceTracker(hmm, max_n_case=EXPERIMENT_CONFIGS[MAX_N_CASE],
                                                 observers=observers)

            caseids = train_event_df['caseid'].unique()
            n_caseids = caseids.shape[0]
            filter_by_conforming_caseids = train_event_df['caseid'].isin(conforming_caseid)
            filtered_train_event_df = train_event_df.loc[~filter_by_conforming_caseids,:]
            n_caseids_train = filtered_train_event_df['caseid'].unique().shape[0]

            info_msg = 'Fitting with {}/{} non-conforming cases'.format(n_caseids_train, n_caseids)
            logger.info(info_msg)

            train_X, train_lengths = event_df_to_hmm_format(filtered_train_event_df)
            start_fit = time.time()
            tracker.hmm.fit(train_X, train_lengths)
            end_fit = time.time()
            took_fit = end_fit - start_fit
            info_msg = 'Training using {} cases took: {:.3f}s'
            info_msg = info_msg.format(n_caseids_train, took_fit)
            logger.info(info_msg)

            # save the 4 key params
            logstartprob_fp = '{}_fold-{}_logstartprob.npy'
            logtranscube_fp = '{}_fold-{}_logtranscube.npy'
            logtranscube_d_fp = '{}_fold-{}_logtranscube_d.npy'
            logemitmat_fp = '{}_fold-{}_logemitmat.npy'
            logemitmat_d_fp = '{}_fold-{}_logemitmat_d.npy'
            confmat_fp = '{}_fold-{}_confmat.npy'

            logstartprob_fp_i = logstartprob_fp.format(net_fname, fold_id)
            logtranscube_fp_i = logtranscube_fp.format(net_fname, fold_id)
            logtranscube_d_fp_i = logtranscube_d_fp.format(net_fname, fold_id)
            logemitmat_fp_i = logemitmat_fp.format(net_fname, fold_id)
            logemitmat_d_fp_i = logemitmat_d_fp.format(net_fname, fold_id)
            confmat_fp_i = confmat_fp.format(net_fname, fold_id)

            logstartprob_fp_i = os.path.join(results_dir_net, logstartprob_fp_i)
            logtranscube_fp_i = os.path.join(results_dir_net, logtranscube_fp_i)
            logtranscube_d_fp_i = os.path.join(results_dir_net, logtranscube_d_fp_i)
            logemitmat_fp_i = os.path.join(results_dir_net, logemitmat_fp_i)
            logemitmat_d_fp_i = os.path.join(results_dir_net, logemitmat_d_fp_i)
            confmat_fp_i = os.path.join(results_dir_net, confmat_fp_i)

            logger.info('Saving learnt parameters')
            with open(logstartprob_fp_i, 'wb') as f:
                np.save(f, tracker.hmm.logstartprob)
            with open(logtranscube_fp_i, 'wb') as f:
                np.save(f, tracker.hmm.logtranscube)
            with open(logtranscube_d_fp_i, 'wb') as f:
                np.save(f, tracker.hmm.logtranscube_d)
            with open(logemitmat_fp_i, 'wb') as f:
                np.save(f, tracker.hmm.logemitmat)
            with open(logemitmat_d_fp_i, 'wb') as f:
                np.save(f, tracker.hmm.logemitmat_d)
            with open(confmat_fp_i, 'wb') as f:
                np.save(f, tracker.hmm.confmat)

            logger.info('Computing the state probability of both train_df and test_df')
            train_conf_rows = list()
            test_conf_rows = list()
            columns = ['caseid', 'case_prefix'] 
            columns += list(state_id_df['state'].values)
            columns += ['emitconf', 'stateconf', 'finalconf', 'injected_distance', 'completeness']

            for row in train_event_df[['caseid', 'activity_id', 'case_prefix']].itertuples(index=False):
                caseid = row.caseid
                event = row.activity_id
                case_prefix = row.case_prefix

                logfwd, finalconf, exception = tracker.replay_event(caseid, event)
                emitconf = conf_observer.emitconf[caseid][-1]
                stateconf = conf_observer.stateconf[caseid][-1]
                injected_dist = injected_dist_rows[-1][2]
                completeness = completeness_rows[-1][2]

                hmmconf_feature = [caseid, case_prefix] + list(logfwd) 
                hmmconf_feature += [emitconf, stateconf, finalconf, injected_dist, completeness]
                train_conf_rows.append(hmmconf_feature)

            for row in test_event_df[['caseid', 'activity_id', 'case_prefix']].itertuples(index=False):
                caseid = row.caseid
                event = row.activity_id
                case_prefix = row.case_prefix

                logfwd, finalconf, exception = tracker.replay_event(caseid, event)
                emitconf = conf_observer.emitconf[caseid][-1]
                stateconf = conf_observer.stateconf[caseid][-1]
                injected_dist = injected_dist_rows[-1][2]
                completeness = completeness_rows[-1][2]

                hmmconf_feature = [caseid, case_prefix] + list(logfwd) 
                hmmconf_feature += [emitconf, stateconf, finalconf, injected_dist, completeness]
                test_conf_rows.append(hmmconf_feature)

            train_results_df = pd.DataFrame.from_records(train_conf_rows, columns=columns)
            test_results_df = pd.DataFrame.from_records(test_conf_rows, columns=columns)
            info_msg = 'Train hmmconf feature df: \n{}'.format(train_results_df.head())
            logger.info(info_msg)
            info_msg = 'Test hmmconf feature df: \n{}'.format(test_results_df.head())
            logger.info(info_msg)

            err_msg = 'hmmconf feature df n_rows: {} != {}: event_df n_rows'
            err_msg_train = err_msg.format(train_results_df.shape[0], train_event_df.shape[0])
            err_msg_test = err_msg.format(test_results_df.shape[0], test_event_df.shape[0])
            assert train_results_df.shape[0] == train_event_df.shape[0], err_msg_train
            assert test_results_df.shape[0] == test_event_df.shape[0], err_msg_test

            # save to store
            if fold_id == 0:
                store['stateid_df'] = state_id_df

            # fold results
            train_df_name = 'train_hmmconf_feature_fold_{}_df'.format(fold_id)
            test_df_name = 'test_hmmconf_feature_fold_{}_df'.format(fold_id)
            store[train_df_name] = train_results_df
            store[test_df_name] = test_results_df

        store.close()
        end_all = time.time()
        took_all = (end_all - start_all) / 60
        logger.info('Took: {:.3f} mins'.format(took_all))
