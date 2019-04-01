import pm_extra, example_data
from pm4py.objects.transition_system import transition_system as ts
import networkx as nx
import numpy as np
import pandas as pd


def rg_to_nx_undirected(rg, map_nodes=False):
    """
    Convert reachability graph to networkx unweighted undirected graph.

    :param: reachability graph
    :type rg: pm4py.TransitionSystem
    :return networkx undirected graph
    """
    assert isinstance(rg, ts.TransitionSystem)

    G = nx.Graph()
    node_map = None

    if map_nodes:
        # map state name to node
        node_map = {key:val for val, key in enumerate(map(lambda state: state.name, rg.states))}
        print(node_map)
        mapped_edges = map(lambda e: (node_map[e.from_state.name], node_map[e.to_state.name]), rg.transitions)
        G.add_edges_from(mapped_edges)
    else:
        G.add_edges_from(map(lambda e: (e.from_state.name, e.to_state.name), rg.transitions))

    return G, node_map


def compute_distance_matrix(G, node_map, as_dataframe=False):
    """
    Assumes that G nodes are integers
    """
    length = nx.all_pairs_dijkstra_path_length(G)
    nb_nodes = len(node_map)
    dist = np.zeros(shape=(nb_nodes, nb_nodes))

    for src, length_dict in length:
        for target in range(nb_nodes):
            dist[src, target] = length_dict[target]

    if as_dataframe:
        nodes = [n[0] for n in sorted(node_map.items(), key=lambda pair: pair[1])]
        dist = pd.DataFrame(dist)
        dist.columns = nodes
        dist['u'] = nodes
        dist.set_index('u', inplace=True)

    return dist


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

