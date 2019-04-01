from pm4py.objects.petri import petrinet as petri
from pm4py.objects.petri import exporter
from pm4py.visualization.transition_system import util

import numpy as np


def build_net1():
    net = petri.PetriNet(name='net1')

    # transitions
    a = petri.PetriNet.Transition('a', label='a')
    b = petri.PetriNet.Transition('b', label='b')
    c = petri.PetriNet.Transition('c', label='c')
    d = petri.PetriNet.Transition('d', label='d')
    e = petri.PetriNet.Transition('e', label='e')
    f = petri.PetriNet.Transition('f', label='f')
    g = petri.PetriNet.Transition('g', label='g')

    trans = [a, b, c, d, e, f, g]

    # places
    p1 = petri.PetriNet.Place('p1')
    p2 = petri.PetriNet.Place('p2')
    p3 = petri.PetriNet.Place('p3')
    p4 = petri.PetriNet.Place('p4')
    p5 = petri.PetriNet.Place('p5')
    p6 = petri.PetriNet.Place('p6')
    p7 = petri.PetriNet.Place('p7')

    places = [p1, p2, p3, p4, p5, p6, p7]

    # arcs
    p1_a = petri.PetriNet.Arc(p1, a)
    p2_b = petri.PetriNet.Arc(p2, b)
    p2_c = petri.PetriNet.Arc(p2, c)
    p3_d = petri.PetriNet.Arc(p3, d)
    p4_e = petri.PetriNet.Arc(p4, e)
    p5_e = petri.PetriNet.Arc(p5, e)
    p6_f = petri.PetriNet.Arc(p6, f)
    p6_g = petri.PetriNet.Arc(p6, g)

    a_p2 = petri.PetriNet.Arc(a, p2)
    a_p3 = petri.PetriNet.Arc(a, p3)
    b_p4 = petri.PetriNet.Arc(b, p4)
    c_p4 = petri.PetriNet.Arc(c, p4)
    d_p5 = petri.PetriNet.Arc(d, p5)
    e_p6 = petri.PetriNet.Arc(e, p6)
    f_p2 = petri.PetriNet.Arc(f, p2)
    f_p3 = petri.PetriNet.Arc(f, p3)
    g_p7 = petri.PetriNet.Arc(g, p7)

    arcs = [
        p1_a, p2_b, p2_c, p3_d, p4_e, p5_e, p6_f, p6_g,
        a_p2, a_p3, b_p4, c_p4, d_p5, e_p6, f_p2, f_p3, g_p7
    ]

    for arc in arcs:
        arc.source.out_arcs.add(arc)
        arc.target.in_arcs.add(arc)

    net.transitions.update(trans)
    net.places.update(places)
    net.arcs.update(arcs)

    init_marking = petri.Marking([p1])
    final_marking = petri.Marking([p7])

    return net, init_marking, final_marking


def net1_state_trans_matrix():
    """
    State transition matrix derived from the reachability graph of net1. Rows
    sum to <= 1. This is a substochastic probability matrix since the final
    state does not transition to any other states. To get the state transition
    probability from state i to state j, you get W_{i,j}.
    """
    W_a = [
        [0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.]
    ]

    W_b = [
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.]
    ]

    W_c = [
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.]
    ]

    W_d = [
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.]
    ]

    W_e = [
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.]
    ]

    W_f = [
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.]
    ]

    W_g = [
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0.]
    ]

    W = np.asarray([
        np.asarray(W_a), np.asarray(W_b), np.asarray(W_c),
        np.asarray(W_d), np.asarray(W_e), np.asarray(W_f),
        np.asarray(W_g)
    ])

    return W


def net1_emission_matrix():
    """
    Emission probability matrix derived from the reachability graph of net1.
    Rows sum to <= 1. This is a substochastic probability matrix since the 
    final state does not emit any activities. To get the emission probability
    of activity a at state i, you get V_{i,a}.
    """
    V = [
        [1., 0., 0., 0., 0., 0., 0.],
        [0., 1./3, 1./3, 1./3, 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., .5, .5, 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., .5, .5],
        [0., 0., 0., 0., 0., 0., 0.]
    ]
    return np.asarray(V)


def net1_conformance_matrix():
    """
    Conformance matrix derived from the reachability graph of net1. For example,
    to check whether if an observation of activity a is conforming if the current
    state is state i, then check C_{a,i}.
    """
    C = [
        [1., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 1., 0., 0., 0.],
        [0., 1., 0., 1., 0., 0., 0.],
        [0., 1., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1., 0.]
    ]
    return np.asarray(C)
