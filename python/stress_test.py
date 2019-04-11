import numpy as np
import pandas as pd
import time, os

import base, lac_setup, pm_extra, tracker, utils
from pm4py.objects.petri.importer import pnml as pnml_importer


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

    hmm = base.HMMConf(startprob, transcube, emitmat, confmat, distmat, 
                       int2state, int2obs, n_states, n_obs, 
                       params='to', verbose=True)
    return hmm


def make_conformance_tracker(hmm):
    return tracker.ConformanceTracker(hmm)


def event_df_to_hmm_format(df):
    lengths = df.groupby(CASEID).count()[ACTIVITY_ID].values
    X = df[[ACTIVITY_ID]].values
    return X, lengths


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
    tracker = make_conformance_tracker(hmm)

    caseids = data_df[CASEID].unique()[-100:]
    to_include = data_df[CASEID].isin(caseids)
    caseids = ['case_34'] # warm start example
    # caseids = data_df[CASEID].unique()[:100]
    n_cases = len(caseids)
    em_to_include = data_df[CASEID].isin(caseids)

    filtered_df = data_df.loc[to_include,:]
    em_filtered_df = data_df.loc[em_to_include,:]

    print('data df shape: {}'.format(filtered_df.shape))

    # EM training
    print('EMing...')
    X, lengths = event_df_to_hmm_format(em_filtered_df)
    fit_start = time.time()
    tracker.hmm.fit(X, lengths)
    fit_end = time.time()
    fit_took = fit_end - fit_start
    logger.info('Training {} cases took: {:.2f}s'.format(n_cases, fit_took))

    # f = open('./results-stress-test.csv', 'w')
    # header = 'caseid,event,conformance,inc_dist,completeness,most_likely_state,state_likelihood'
    # print(header, file=f)

    i = 0
    limit = 1000
    for row in em_filtered_df.itertuples(index=False):
        caseid = row.caseid
        event = row.activity_id
        act = row.activity

        score = tracker.replay_event(caseid, event)
        conf_arr = score[0]
        most_likely_state = score[1]
        likelihood_mode = score[2]
        complete = score[3]
        exp_inc_dist = score[4]
        mode_dist = score[5]
        sum_dist = score[6]
        sum_mode_dist = score[7]

        msg = '{caseid} replay {event:<11}:    ' \
            'conf: {conf:.2f}, compl: {compl:.2f}, ' \
            'inc_dist: {inc_dist:.2f}, mode_dist: {mode_dist:.2f}, ' \
            'sum_dist: {sum_dist:.2f}, sum_mode_dist: {sum_mode_dist:.2f}, ' \
            '{state}, {like:.2f}'
        msg = msg.format(caseid=caseid, event=act, conf=conf_arr[2],
                         compl=complete, inc_dist=exp_inc_dist,
                         mode_dist=mode_dist, sum_dist=sum_dist,
                         sum_mode_dist=sum_mode_dist, 
                         state=most_likely_state, like=likelihood_mode)
        print(msg)
        # print(caseid, act, conf[2], exp_inc_dist, complete, state, mode, file=f, sep=',')

        # time.sleep(0.1)

        # i += 1
        # if i > limit:
        #     break

    # f.close()

    end = time.time()
    took = end - start
    print('Stress test took {:.2f}s'.format(took))
