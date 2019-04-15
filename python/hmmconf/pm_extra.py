"""
Extra functionalities for process mining related stuff
======================================================


"""

import re, warnings
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pm4py.objects.transition_system import transition_system as ts
from pm4py.objects.petri import semantics
from pm4py.objects.petri import petrinet as petri


__all__ = [
    'build_reachability_graph',
    'get_init_marking',
    'connect_inv_markings',
    'collapse_inv_trans'
]


MAX_RG_STATE = 1e6


plt.switch_backend('TkAgg')
# print('matplotlib using backend: {}'.format(plt.get_backend()))


class Transition(object):
    def __init__(self, name, from_state, to_state, data=None):
        self.name = name
        self.from_state = from_state
        self.to_state = to_state
        self.data = dict() if data is None else data

    def __repr__(self):
        return str(self.name)


ts.TransitionSystem.Transition = Transition


def default_staterep(name):
    return re.sub(r'\W+', '_', name[2:-2])


def default_staterep_to_marking(staterep, place_map):
    place_groups = re.split(r'([a-zA-Z0-9]+_[0-9]+)', staterep)
    places = []
    # print('Splitted groups: {}'.format(place_groups))
    for grp in place_groups:
        if grp == '' or grp == '_':
            continue
        place_name, nb_tokens = grp.split('_')
        # print('Group: {}, place name: {}, no. of tokens: {}'.format(grp, place_name, nb_tokens))
        place = place_map[place_name]
        places.extend([place for _ in range(int(nb_tokens))])
    marking = petri.Marking(places)
    return marking


def add_arc_from_to(name, fr, to, gr, data=None):
    t = ts.TransitionSystem.Transition(name, fr, to, data) # a little hack due to in_state and out_state being sets and without setter methods before pm4py version 1.1.3
    gr.transitions.add(t)
    fr.outgoing.add(t)
    to.incoming.add(t)
    return t


def build_reachability_graph(net, init_marking, is_inv, staterep=default_staterep):
    """
    Build reachability graph and keep track of to and from states connected by 
    invisible transitions.

    :param net: the petrinet 
    :param init_marking: initial marking
    :param is_inv: function that indicate if a transition is invisible
    :type is_inv: function
    :staterep: function that gives a state a string representation
    :return reachability graph, list of (from_state, inv_tran, to_state)
    """

    # BFS with queue
    mark_queue = [init_marking]

    rg = ts.TransitionSystem(name='Reachability graph of {}'.format(net.name))
    inv_states = list()

    init_state = ts.TransitionSystem.State(staterep(repr(init_marking)))
    init_state.data['disc'] = 0
    rg.states.add(init_state)

    # mapping visited states to marking
    mark_to_state = dict()
    mark_to_state[init_marking] = init_state

    while mark_queue and len(rg.states) < MAX_RG_STATE:
        cur_mark = mark_queue.pop()
        cur_state = mark_to_state[cur_mark]
        enabled_trans = list(semantics.enabled_transitions(net, cur_mark))

        # workout the transition arc weight
        n_vis = len(list(map(lambda t: not is_inv(t), enabled_trans)))
        weight = 1. / n_vis if n_vis > 0 else 0

        for t in enabled_trans:
            next_mark = semantics.execute(t, net, cur_mark)
            next_state = mark_to_state.get(next_mark, None)

            if next_state is None:
                next_state = ts.TransitionSystem.State(staterep(repr(next_mark)))
                # discovered one step away from parent node
                next_state.data['disc'] = cur_state.data['disc'] + 1
                rg.states.add(next_state)
                mark_to_state[next_mark] = next_state
                mark_queue.append(next_mark)

            # doesnt matter that invisible transitions also get weight since
            # they will be removed ultimately as well
            data = {'weight': weight} 
            t_label = t.label if t.name is not None else None
            rg_t = add_arc_from_to(t_label, cur_state, next_state, rg, data)

            if is_inv(t):
                inv_states.append((cur_state, rg_t, next_state))

    if mark_queue:
        msg = 'Computation of reachability graph surpass the max number of states: {}'.format(MAX_RG_STATE)
        warnings.warn(msg, category=UserWarning)
     
    return rg, inv_states


def get_init_marking(rg):
    # under assumption of workflow net
    init = list(filter(lambda s: len(s.incoming) == 0, rg.states))
    assert len(init) == 1, 'Violate workflow net assumption, init: {}'.format(init)
    return init[0]


def connect_inv_markings(rg, inv_states, is_inv):
    init = get_init_marking(rg)
    node_q = [init] 
    visited = set()
    visited.add(init)

    def add_inv_states(state, inv_states, _buffer, visited, is_inv):
        n_inv = 0
        for t in state.outgoing:
            if is_inv(t) and (state, t.to_state) not in visited:
                n_inv += 1
                inv_states.append(t.to_state)
                _buffer.append(t.to_state)
                visited.add((state, t.to_state))
        return n_inv

    while len(node_q) > 0:
        cur_state = node_q.pop(0)

        to_connect = list()
        for t in cur_state.outgoing:
            if t.to_state not in visited:
                visited.add(t.to_state)
                node_q.append(t.to_state)

            # no need to process an invisible transition
            if is_inv(t):
                continue

            n_inv = 0
            to_connect_t = list()
            visited_t = set()
            _buffer = list()

            n_inv += add_inv_states(t.to_state, to_connect_t, 
                                    _buffer, visited_t, is_inv)

            while len(_buffer) > 0:
                # keep add adjacent nodes 
                s = _buffer.pop(0)
                n_inv += add_inv_states(s, to_connect_t, _buffer, 
                                        visited_t, is_inv)

            if n_inv > 0:
                # diminish t's weight to connect to next states
                w = t.data['weight'] / n_inv
                t.data['weight'] = w
                for state in to_connect_t:
                    data = {'weight': w}
                    to_connect.append((t.name, cur_state, state, rg, data))
        
        for name, from_state, to_state, rg, data in to_connect:
            add_arc_from_to(name, from_state, to_state, rg, data)

    # removal of invisible transition
    for in_state, inv_tran, out_state in inv_states:
        if in_state.name == init.name:
            continue
        in_state.outgoing.remove(inv_tran)
        out_state.incoming.remove(inv_tran)
        rg.transitions.remove(inv_tran)


def collapse_inv_trans(rg, inv_states):
    """Collapse transition edges in reachability graph that correspond to invisible transitions and modify the adjacent edge probability weight accordingly.

    :param rg: reachability graph
    :param inv_states array_like: ordered list of tuples (in_state, inv_tran, out_state) such that earlier elements are ones were explored first during the breadth-first search that created the reachability graph.
    """
    for _, inv_tran, _ in inv_states[::-1]:
        in_state = inv_tran.from_state
        out_state = inv_tran.to_state
        # print('In state: {}'.format(in_state))
        # print('Transition: {}'.format(inv_tran))
        # print('Out state: {}'.format(out_state))
        # connect all incoming arcs to in_state to out_state
        inv_tran_weight = inv_tran.data['weight']
        in_state.outgoing.remove(inv_tran)
        out_state.incoming.remove(inv_tran)

        for out_tran in out_state.outgoing:
            # make new transition that connect in_state to out_tran.out_state
            data = {'weight': out_tran.data['weight'] * inv_tran_weight}
            add_arc_from_to(out_tran.name, in_state, out_tran.to_state, rg, data)

        # remove out_state if it is no longer a reachable marking
        if len(out_state.incoming) == 0:
            rg.states.remove(out_state)
            for out_tran in out_state.outgoing:
                rg.transitions.remove(out_tran)
                out_tran.to_state.incoming.remove(out_tran)

        # remove in_state if it no longer has any outgoing arcs
        if len(in_state.outgoing) == 0 and len(in_state.incoming) == 0:
            rg.states.remove(in_state)

        rg.transitions.remove(inv_tran)


def draw_undirected(G, node_map):
    pos = nx.spring_layout(G)
    
    # nodes
    node_color = 'lightblue'
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_color)

    # edges
    edges = [(u, v) for (u, v, d) in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    # create legend from node map
    handles = list()
    for key, val in node_map.items():
        patch = mpatches.Patch(color=node_color, label='{}:{}'.format(key, val))
        handles.append(patch)

    plt.legend(handles=handles)


if __name__ == '__main__':
    print('Testing build_reachability_graph...')

    from pm4py.objects.petri import exporter 
    from pm4py.visualization.transition_system import util

    net = petri.PetriNet()

    # transitions
    a = petri.PetriNet.Transition('a', label='Activity A')
    b = petri.PetriNet.Transition('b', label='Activity B')
    c = petri.PetriNet.Transition('c', label='Activity C')
    d = petri.PetriNet.Transition('d', label='Activity D')
    e = petri.PetriNet.Transition('e', label='Activity E')
    f = petri.PetriNet.Transition('f', label='Activity F')
    g = petri.PetriNet.Transition('g', label='Activity G')
    h = petri.PetriNet.Transition('h', label='Activity H')
    inv0 = petri.PetriNet.Transition('inv0', label=None)
    inv1 = petri.PetriNet.Transition('inv1', label=None)

    trans = [a, b, c, d, e, f, g, h, inv0, inv1]

    # places
    p0 = petri.PetriNet.Place('p0')
    p1 = petri.PetriNet.Place('p1')
    p2 = petri.PetriNet.Place('p2')
    p3 = petri.PetriNet.Place('p3')
    p4 = petri.PetriNet.Place('p4')
    p5 = petri.PetriNet.Place('p5')
    p6 = petri.PetriNet.Place('p6')
    p7 = petri.PetriNet.Place('p7')
    p8 = petri.PetriNet.Place('p8')
    p9 = petri.PetriNet.Place('p9')
    p10 = petri.PetriNet.Place('p10')
    p11 = petri.PetriNet.Place('p11')

    places = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]

    # arcs
    p0_a = petri.PetriNet.Arc(p0, a)
    p1_c = petri.PetriNet.Arc(p1, c)
    p2_inv0 = petri.PetriNet.Arc(p2, inv0)
    p3_e = petri.PetriNet.Arc(p3, e)
    p4_f = petri.PetriNet.Arc(p4, f)
    p5_g = petri.PetriNet.Arc(p5, g)
    p6_inv1 = petri.PetriNet.Arc(p6, inv1)
    p7_inv1 = petri.PetriNet.Arc(p7, inv1)
    p8_inv1 = petri.PetriNet.Arc(p8, inv1)
    p9_d = petri.PetriNet.Arc(p9, d)
    p10_h = petri.PetriNet.Arc(p10, h)
    p10_b = petri.PetriNet.Arc(p10, b)

    a_p1 = petri.PetriNet.Arc(a, p1)
    c_p2 = petri.PetriNet.Arc(c, p2)
    inv0_p3 = petri.PetriNet.Arc(inv0, p3)
    inv0_p4 = petri.PetriNet.Arc(inv0, p4)
    inv0_p5 = petri.PetriNet.Arc(inv0, p5)
    e_p6 = petri.PetriNet.Arc(e, p6)
    f_p7 = petri.PetriNet.Arc(f, p7)
    g_p8 = petri.PetriNet.Arc(g, p8)
    inv1_p9 = petri.PetriNet.Arc(inv1, p9)
    d_p10 = petri.PetriNet.Arc(d, p10)
    h_p1 = petri.PetriNet.Arc(h, p1)
    b_p11 = petri.PetriNet.Arc(b, p11)

    arcs = [
        p0_a, p1_c, p2_inv0, p3_e, p4_f, p5_g, p6_inv1, p7_inv1, 
        p8_inv1, p9_d, p10_h, p10_b, a_p1, c_p2, inv0_p3, inv0_p4,
        inv0_p5, e_p6, f_p7, g_p8, inv1_p9, d_p10, h_p1, b_p11
    ]

    for arc in arcs:
        arc.source.out_arcs.add(arc)
        arc.target.in_arcs.add(arc)

    net.transitions.update(trans)
    net.places.update(places)
    net.arcs.update(arcs)

    init_marking = petri.Marking([p0])
    final_marking = petri.Marking([p11])

    # out_fp = './test-net.pnml'
    # exporter.pnml.export_net(net, init_marking, out_fp, final_marking=final_marking)

    # build the reachability graph
    is_inv = lambda t: t.label is None
    rg, inv_states = build_reachability_graph(net, init_marking, is_inv)

    collapse_inv_trans(rg, inv_states)

#    out_fp = './image/test-rg.dot'
#    dot_gr = util.visualize_graphviz.visualize(rg)
#    dot_gr.save(out_fp, '.')
