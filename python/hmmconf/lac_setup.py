from . import pm_extra, example_data, utils
from pm4py.objects.transition_system import transition_system as ts
import networkx as nx
import numpy as np
import pandas as pd


__all__ = [
    'rg_to_nx_undirected',
    'compute_distance_matrix',
    'compute_state_trans_cube',
    'compute_emission_mat',
    'compute_conformance_mat',
    'compute_startprob'
]


logger = utils.make_logger(__file__)


def rg_to_nx_undirected(rg, map_nodes=False):
    """
    Convert reachability graph to networkx unweighted undirected graph.

    :param rg: reachability graph
    :type rg: pm4py.TransitionSystem
    :param map_nodes bool, optional: if ``True`` then map graph node names to int
    :return networkx undirected graph, node2int mapping
    """
    assert isinstance(rg, ts.TransitionSystem)

    G = nx.Graph()
    node_map = None

    if map_nodes:
        # sort states by discovery time then by their name to break ties
        sorted_states = sorted(list(rg.states), key=lambda s: (s.data['disc'], s.name))
        # map state name to node
        node_map = {key:val for val, key in enumerate(map(lambda state: state.name, sorted_states))}
        # print(node_map)
        mapped_edges = map(lambda e: (node_map[e.from_state.name], node_map[e.to_state.name]), rg.transitions)
        G.add_edges_from(mapped_edges)
    else:
        G.add_edges_from(map(lambda e: (e.from_state.name, e.to_state.name), rg.transitions))

    return G, node_map


def compute_distance_matrix(G, node2int, as_dataframe=False):
    """
    Assumes that G nodes are integers

    :param G: graph
    :param node2int dict: mapping from node name to int
    :return state distance matrix
    """
    length = nx.all_pairs_dijkstra_path_length(G)
    nb_nodes = len(node2int)
    dist = np.zeros(shape=(nb_nodes, nb_nodes))

    for src, length_dict in length:
        for target in range(nb_nodes):
            dist[src, target] = length_dict[target]

    dist_min, dist_max, dist_mean, dist_std = np.min(dist), np.max(dist), np.mean(dist), np.std(dist)
    logger.info('dist mat min: {}, max: {}, mean: {:.2f}, std: {:.2f}'.format(dist_min, 
                                                                              dist_max,
                                                                              dist_mean,
                                                                              dist_std))
    if as_dataframe:
        nodes = [n[1] for n in sorted(node2int.items(), key=lambda pair: pair[1])]
        dist = pd.DataFrame(dist)
        dist.columns = nodes
        dist['u'] = nodes
        dist.set_index('u', inplace=True)

    return dist


def compute_state_trans_cube(rg, state2int, obs2int, n_obs, n_states):
    """Computes the state transition probability cube from reachability graph.

    :param rg: reachability graph
    :param state2int dict: mapping from state name to integer
    :param obs2int dict: mapping from observation to integer
    :param n_obs int: number of observations
    :param n_states int: number of states
    :return: state transition probability cube
    """
    cube = np.zeros((n_obs, n_states, n_states))
    
    for in_state in rg.states:
        in_state_ind = state2int[in_state.name]

        for tran in in_state.outgoing:
            obs_ind = obs2int[tran.name]
            out_state_ind = state2int[tran.to_state.name]

            cube[obs_ind, in_state_ind, out_state_ind] += tran.data['weight']

    utils.normalize(cube, axis=2)
    return cube

def compute_emission_mat(rg, state2int, obs2int, n_obs, n_states):
    """Computes the emission probability matrix from reachability graph.

    :param rg: reachability graph
    :param state2int dict: mapping from state name to integer
    :param obs2int dict: mapping from observation to integer
    :param n_obs int: number of observations
    :param n_states int: number of states
    :return: emission probability matrix
    """
    emitmat = np.zeros((n_states, n_obs))

    for in_state in rg.states:
        in_state_ind = state2int[in_state.name]

        for tran in in_state.outgoing:
            obs_ind = obs2int[tran.name]
            emitmat[in_state_ind, obs_ind] += tran.data['weight']

    utils.normalize(emitmat, axis=1)
    return emitmat


def compute_conformance_mat(emitmat):
    """Computes the conformance matrix which is boolean version of the emission probability
    matrix such that C[i,j] is ``True`` if emitmat[i,j] > 0, and is ``False`` otherwise.

    :param emitmat array_like: emission probability matrix
    :return: conformance matrix
    """
    return (emitmat > 0).astype(np.int).T


def compute_startprob(rg, state2int, n_states, is_inv):
    """Computes the initial state estimation which is a one-hot vector with point mass at the 
    initial marking state.

    :param rg: reachability graph
    :param state2int dict: mapping from state name to integer
    :param n_states int: number of states
    """
    # init = list(filter(lambda s: len(s.incoming) == 0, rg.states))
    init = pm_extra.get_init_marking(rg)

    # if len(init) != 1:
    #     raise ValueError('Number of states with 0 incoming transitions: {}'.format(len(init)))

    init_inds = set()
    init_marks = set()
    init_inds.add(state2int[init.name])
    init_marks.add(init.name)

    _buffer = set()

    for t in init.outgoing:
        if is_inv(t):
            init_inds.add(state2int[t.to_state.name])
            init_marks.add(t.to_state.name)
            _buffer.add(t.to_state)

    while len(_buffer) > 0:
        state = _buffer.pop()
        for t in state.outgoing:
            if is_inv(t):
                init_inds.add(state2int[t.to_state.name])
                init_marks.add(t.to_state.name)
                _buffer.add(t.to_state)

    weight = 1. / len(init_inds)
    
    startprob = np.zeros((1, n_states))
    for ind in init_inds:
        startprob[0, ind] = weight

    logger.info('Initial states: {}'.format(init_marks))

    return startprob


if __name__ == '__main__':
    from pm4py.visualization.petrinet.common import visualize
    from pm4py.visualization.transition_system import util

    net, init_marking, final_marking = example_data.net1()
    place_map = {p.name:p for p in net.places}
    
#    out_fp = './image/net1.dot'
#    parameters = {
#        'set_rankdir_lr': True
#    }
#    dot_net = visualize.apply(net, initial_marking=init_marking, 
#                              final_marking=final_marking, parameters=parameters)
#    dot_net.save(out_fp, '.')
    
    is_inv = lambda t: t.name == ''
    rg, inv_states = pm_extra.build_reachability_graph(net, init_marking, is_inv)

    out_fp = './image/net1-rg.dot'
    dot_rg = util.visualize_graphviz.visualize(rg)
    dot_rg.save(out_fp, '.')

    G, node_map = rg_to_nx_undirected(rg, map_nodes=True)
    dist_df = compute_distance_matrix(G, node_map, as_dataframe=True)
    print(dist_df)

    marking_str = dist_df.columns[0]
    print('Checking retrieval of marking: {}'.format(marking_str))
    marking = pm_extra.default_staterep_to_marking(marking_str, place_map)
    print('{}'.format(marking))

