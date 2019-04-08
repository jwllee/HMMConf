import numpy as np
import pandas as pd
import time, os

import base, lac_setup, pm_extra, tracker
from pm4py.objects.petri.importer import pnml as pnml_importer


MODEL_FP = os.path.join('..', 'data', 'BPM2018', 'stress-test', 'model.pnml')
DATA_FP = os.path.join('..', 'data', 'BPM2018', 'stress-test', 'filtered-stream.csv')

ACTIVITY = 'activity'
ACTIVITY_ID = 'activity_id'
CASEID = 'caseid'


np.set_printoptions(precision=2)


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
    pm_extra.collapse_inv_trans(rg, inv_states)
    return rg


def setup_hmm(rg):
    G, node_map = lac_setup.rg_to_nx_undirected(rg, map_nodes=True)
    dist_df = lac_setup.compute_distance_matrix(G, node_map, as_dataframe=True)
    distmat = dist_df.values
    # print('Distance df: \n{}'.format(dist_df))

    obsmap = {t.name: int(t.name) for t in rg.transitions}
    int2state = {val:key for key, val in node_map.items()}
    int2obs = {val:key for key, val in obsmap.items()}
    n_obs = len(obsmap)
    n_states = len(node_map)

    transcube = lac_setup.compute_state_trans_cube(rg, node_map, obsmap, n_obs, n_states)
    emitmat = lac_setup.compute_emission_mat(rg, node_map, obsmap, n_obs, n_states)
    confmat = lac_setup.compute_conformance_mat(emitmat)
    startprob = lac_setup.compute_startprob(rg, node_map, n_states)

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

    caseids = data_df[CASEID].unique()[:100]
    to_include = data_df[CASEID].isin(caseids)
    caseids = ['case_34'] # warm start example
    em_to_include = data_df[CASEID].isin(caseids)

    em_filtered_df = data_df.loc[em_to_include,:]
    filtered_df = data_df.loc[to_include,:]

    print('data df shape: {}'.format(filtered_df.shape))

    # EM training
    print('EMing...')
    X, lengths = event_df_to_hmm_format(em_filtered_df)
    tracker.hmm.fit(X, lengths)

    i = 0
    limit = 1000
    for row in filtered_df.itertuples(index=False):
        caseid = row.caseid
        event = row.activity_id
        act = row.activity

        score, state = tracker.replay_event(caseid, event)
        print('{} replay {:<11}: {:.2f}, {}'.format(caseid, act, score[2], state))

        # time.sleep(0.1)

        i += 1
        if i > limit:
            break

    end = time.time()
    took = end - start
    print('Stress test took {:.2f}s'.format(took))
