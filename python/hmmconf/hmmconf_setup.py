from copy import copy
from . import pm_extra, example_data, utils
from pm4py.objects.transition_system import transition_system as ts
from pm4py.objects.petri import semantics


from hmmconf import preprocess


import networkx as nx
import numpy as np
import pandas as pd


__all__ = [
    'rg_to_nx_undirected',
    'compute_distance_matrix',
    'compute_state_trans_cube',
    'compute_emission_mat',
    'compute_conformance_mat',
    'compute_startprob',
    'estimate_transcube',
    'estimate_emitmat',
    'get_counts_from_log',
    'get_counts_from_case',
    'get_pseudo_counts_transcube',
    'get_pseudo_counts_emitmat',
    'compute_confmat',
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


def get_endpoint_vis_trans(v, is_inv):
    """Get all visible transitions enabled by a descendant node reachable via
    invisible transitions.
    """
    vis_trans = set()
    to_explore = [v]
    visited = set()
    visited.add(v)

    while len(to_explore) > 0:
        n = to_explore.pop(0)
        for t in n.outgoing:
            if is_inv(t):
                if t.to_state not in visited:
                    to_explore.append(t.to_state)
                    visited.add(t.to_state)
            else:
                vis_trans.add(t)

    return vis_trans


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


def compute_distmat(rg, state2int, is_inv):
    """Traverse the reachability graph to compute distance matrix.
    """
    def compute_shortest_paths(v, distmat, state2int, is_inv):
        """Computes shortest path distance from state v to all reachable states in place
        """
        v_ind = state2int[v.name]
        # self distance is 0 by default
        distmat[v_ind,v_ind] = 0
        
        # perform BFS
        node_q = [
            (None, v) # each node consists of state parent and state
        ]
        visited = set()
        visited.add(v)
        n_states = len(state2int)

        while len(node_q) > 0:
            p, n = node_q.pop(0)
            n_ind = state2int[n.name]
            p_ind = state2int[p.name] if p is not None else None
            # get distance from source node v to c.parent
            # since it's a BFS, we must have had explored its parent already
            dist_p = distmat[v_ind,n_ind]

            # info_msg = 'Distance between {} to {} = {}'
            # info_msg = info_msg.format(v.name, n.name, dist_p)
            # logger.info(info_msg)

            for t in n.outgoing:
                c = t.to_state
                c_ind = state2int[c.name]

                dist = dist_p + abs(is_inv(t) - 1)

                if distmat[v_ind,c_ind] >= 0:
                    # info_msg = 'min({}, {}) for {} -> {}'
                    # info_msg = info_msg.format(distmat[v_ind,c_ind], dist, v.name, c.name)
                    # logger.info(info_msg)
                    distmat[v_ind,c_ind] = min(distmat[v_ind,c_ind], dist)
                else:
                    distmat[v_ind,c_ind] = dist

                # info_msg = 'Distance between {} to {} (parent: {}) = {}'
                # info_msg = info_msg.format(v.name, c.name, n.name, distmat[v_ind,c_ind])
                # logger.info(info_msg)

                if not c in visited:
                    visited.add(c)
                    node_q.append((n, c))

    def rg2g(rg, state2int, is_inv):
        g = nx.Graph()
        edges = []
        for t in rg.transitions:
            weight = abs(is_inv(t) - 1)
            edge = (state2int[t.from_state.name], state2int[t.to_state.name], weight)
            edges.append(edge)
        g.add_weighted_edges_from(edges)
        return g

    n_states = len(state2int)
    distmat = np.full((n_states, n_states), -1)

    for s in rg.states:
        compute_shortest_paths(s, distmat, state2int, is_inv)

    # there might be pairs of nodes that are not reachable from each other in the directed graph
    # convert reachability graph into undirected graph to compute node distances for these node pairs
    has_uncomputed = (distmat == -1).any()

    # return distmat if all node distances have been computed without resorting to undirected graph conversion
    if not has_uncomputed:
        return distmat

    g = rg2g(rg, state2int, is_inv)
    length = nx.all_pairs_dijkstra_path_length(g)

    for src, length_dict in length:
        for target in state2int.values():
            if distmat[src, target] == -1:
                distmat[src, target] = length_dict[target]

    return distmat


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


def compute_conformance_mat(rg, state2int, obs2int, n_obs, n_states):
    """Computes the conformance matrix which is boolean version of the emission probability
    matrix such that C[i,j] is ``True`` if emitmat[i,j] > 0, and is ``False`` otherwise.

    :param emitmat array_like: emission probability matrix
    :return: conformance matrix
    """
    emitmat = np.zeros((n_states, n_obs))

    for in_state in rg.states:
        in_state_ind = state2int[in_state.name]

        for tran in in_state.outgoing:
            obs_ind = obs2int[tran.name]
            emitmat[in_state_ind, obs_ind] += tran.data['weight']

    utils.normalize(emitmat, axis=1)
    return (emitmat > 0).astype(np.float).T


def compute_confmat(rg, init, is_inv, state2int=None, obs2int=None):
    """Computes a conformance matrix so that confmat[m,a] = 1 iff there is a path of invisible 
    transitions from marking m that leads to the enabling of the corresponding transition or 
    marking m is forward propagating the enabled transition through a path of invisible transitions.
    """
    n_states, n_obs = len(state2int), len(obs2int)
    confmat = np.zeros((n_obs, n_states))
    node_q = [(None, init)]
    visited = set()
    visited.add(init)

    while len(node_q) > 0:
        t, n = node_q.pop(0)
        t_ind = obs2int[t.name] if t and t.name else None
        n_ind = state2int[n.name]

        # info_msg = 'Visiting node {}'
        # info_msg = info_msg.format(n)
        # logger.info(info_msg)

        children = [(t_c, t_c.to_state) for t_c in n.outgoing]
        for t_c, c in children:
            # info_msg = 'Child: {} with transition: {}'
            # info_msg = info_msg.format(c, t_c)
            # logger.info(info_msg)

            c_ind = state2int[c.name]

            if not is_inv(t_c):
                t_c_ind = obs2int[t_c.name]
                confmat[t_c_ind,n_ind] = 1
            else:
                # forward propagating
                endpoint_vis_trans = get_endpoint_vis_trans(c, is_inv)
                for e in endpoint_vis_trans:
                    e_ind = obs2int[e.name]
                    confmat[e_ind,n_ind] = 1

            # adding unvisited child to node queue
            if c not in visited:
                node_q.append((t_c, c))
                visited.add(c)

    return confmat


def compute_startprob(rg, state2int, is_inv):
    """Computes the initial state estimation which is probability mass spread 
    uniformly over the enabling states reachable via invisible transitions from
    the initial marking of the Petrinet model.

    :param rg: reachability graph
    :param state2int dict: mapping from state name to integer
    :param n_states int: number of states
    """
    # init = list(filter(lambda s: len(s.incoming) == 0, rg.states))
    n_states = len(state2int)
    init = pm_extra.get_init_marking(rg)
    endpoint_vis_trans = get_endpoint_vis_trans(init, is_inv)

    init_states = set()
    init_states.add(init)

    for t in endpoint_vis_trans:
        init_states.add(t.from_state)

    logger.info('Initial states: {}'.format(init_states))

    startprob = np.zeros(n_states,)
    for s in init_states:
        ind = state2int[s.name]
        startprob[ind] = 1
    startprob /= startprob.sum()

    return startprob

    # if len(init) != 1:
    #     raise ValueError('Number of states with 0 incoming transitions: {}'.format(len(init)))

#    init_inds = set()
#    init_marks = set()
#    init_inds.add(state2int[init.name])
#    init_marks.add(init.name)
#
#    _buffer = set()
#
#    for t in init.outgoing:
#        if is_inv(t):
#            init_inds.add(state2int[t.to_state.name])
#            init_marks.add(t.to_state.name)
#            _buffer.add(t.to_state)
#
#    while len(_buffer) > 0:
#        state = _buffer.pop()
#        for t in state.outgoing:
#            if is_inv(t):
#                init_inds.add(state2int[t.to_state.name])
#                init_marks.add(t.to_state.name)
#                _buffer.add(t.to_state)
#
#    weight = 1. / len(init_inds)
#    
#    startprob = np.zeros((1, n_states))
#    for ind in init_inds:
#        startprob[0, ind] = weight
#
#    logger.info('Initial states: {}'.format(init_marks))
#
#    return startprob


def get_pseudo_counts_transcube(rg, init, is_inv, state2int, obs2int, multiplier=1.):
    """Traverse the reachability graph to accumulate pseudo counts as prior distribution.
    """
    n_states, n_obs = len(state2int), len(obs2int)
    trans_count = np.zeros((n_obs, n_states, n_states))
    node_q = [(None, init)]
    visited = set()
    visited.add(init)

    while len(node_q) > 0:
        t, n = node_q.pop(0)
        t_ind = obs2int[t.name] if t and t.name else None
        n_ind = state2int[n.name]

        # info_msg = 'Visiting node {}'
        # info_msg = info_msg.format(n)
        # logger.info(info_msg)

        children = [(t_c, t_c.to_state) for t_c in n.outgoing]
        for t_c, c in children:
            # info_msg = 'Child: {} with transition: {}'
            # info_msg = info_msg.format(c, t_c)
            # logger.info(info_msg)

            c_ind = state2int[c.name]

            if not is_inv(t_c):
                t_c_ind = obs2int[t_c.name]
                trans_count[t_c_ind,n_ind,c_ind] += 1
            else:
                # forward propagating
                endpoint_vis_trans = get_endpoint_vis_trans(c, is_inv)
                for e in endpoint_vis_trans:
                    e_ind = obs2int[e.name]
                    to_state = e.to_state
                    to_state_ind = state2int[to_state.name]

                    trans_count[e_ind,n_ind,to_state_ind] += 1

            # adding unvisited child to node queue
            if c not in visited:
                node_q.append((t_c, c))
                visited.add(c)

    trans_count *= multiplier

    return trans_count


def get_pseudo_counts_emitmat(rg, init, is_inv, state2int, obs2int, multiplier=1.):
    """Traverse reachability to accumulate pseudo counts as prior distribution.
    """
    n_states, n_obs = len(state2int), len(obs2int)
    emit_count = np.zeros((n_states, n_obs))
    node_q = [(None, init)]
    visited = set()
    visited.add(init)

    while len(node_q) > 0:
        t, n = node_q.pop(0)
        t_ind = obs2int[t.name] if t and t.name else None
        n_ind = state2int[n.name]

        # info_msg = 'Visiting node {}'
        # info_msg = info_msg.format(n)
        # logger.info(info_msg)

        children = [(t_c, t_c.to_state) for t_c in n.outgoing]
        for t_c, c in children:
            # info_msg = 'Child: {} with transition: {}'
            # info_msg = info_msg.format(c, t_c)
            # logger.info(info_msg)

            c_ind = state2int[c.name]

            if not is_inv(t_c):
                t_c_ind = obs2int[t_c.name]
                emit_count[n_ind,t_c_ind] += 1
            else:
                # forward propagating
                endpoint_vis_trans = get_endpoint_vis_trans(c, is_inv)
                for e in endpoint_vis_trans:
                    e_ind = obs2int[e.name]
                    emit_count[n_ind,e_ind] += 1

            # adding unvisited child to node queue
            if c not in visited:
                node_q.append((t_c, c))
                visited.add(c)

    emit_count *= multiplier

    return emit_count


def get_counts_from_case(case, state2int, obs2int, net, init_marking, final_marking, is_inv):
    conforming = False
    n_obs = len(obs2int)
    n_states = len(state2int)

    trans_count = np.zeros((n_obs, n_states, n_states))
    emit_count = np.zeros((n_states, n_obs))

    marking_seq = preprocess.get_marking_sequence(case, net, init_marking, final_marking, is_inv)
    
    if not marking_seq:
        return trans_count, emit_count, conforming

    conforming = True
    last_obs, last_marking = marking_seq.pop(0)
    last_marking_str = pm_extra.marking2str(last_marking)
    emit_count[state2int[last_marking_str], obs2int[last_obs]] += 1

    while marking_seq:
        obs, marking = marking_seq.pop(0)
        marking_str = pm_extra.marking2str(marking)
        # obs might be None for the final marking
        if obs:
            emit_count[state2int[marking_str], obs2int[obs]] += 1
        trans_count[obs2int[last_obs], state2int[last_marking_str], state2int[marking_str]] += 1
        last_obs = obs
        last_marking = marking
        last_marking_str = marking_str

    return trans_count, emit_count, conforming


def get_counts_from_log(cases, state2int, obs2int, net, init_marking, final_marking, is_inv):
    if len(cases) == 0:
        raise ValueError

    conforming_caseids = list()
    caseid, case = cases.pop(0)
    trans_count, emit_count, conforming = get_counts_from_case(case, state2int, obs2int, 
                                                   net, init_marking, final_marking, is_inv)
    if conforming:
        conforming_caseids.append(caseid)

    while cases:
        caseid, case = cases.pop(0)
        trans_count_i, emit_count_i, conforming = get_counts_from_case(case, state2int, obs2int,
                                                           net, init_marking, final_marking, is_inv)
        trans_count += trans_count_i
        emit_count += emit_count_i

        if conforming:
            conforming_caseids.append(caseid)

    return trans_count, emit_count, conforming_caseids


def estimate_transcube(trans_count, trans_pseudo=None):
    transcube = trans_count + trans_pseudo if trans_pseudo is not None else trans_count
    utils.normalize(transcube, axis=2)
    return transcube


def estimate_emitmat(emit_count, emit_pseudo=None):
    emitmat = emit_count + emit_pseudo if emit_pseudo is not None else emit_count
    utils.normalize(emitmat, axis=1)
    return emitmat


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

