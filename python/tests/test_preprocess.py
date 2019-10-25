import time
from copy import copy
from pytest import fixture

from hmmconf import preprocess
from pm4py.objects.petri.petrinet import PetriNet, Marking
from pm4py.objects.petri import semantics


@fixture
def ex_petrinet():
    net = PetriNet()

    # transitions
    a = PetriNet.Transition('a', label='a')
    b = PetriNet.Transition('b', label='b')
    c = PetriNet.Transition('c', label='c')
    d = PetriNet.Transition('d', label='d')
    e = PetriNet.Transition('e', label='e')
    f = PetriNet.Transition('f', label='f')
    g = PetriNet.Transition('g', label='g')
    h = PetriNet.Transition('h', label='h')
    inv0 = PetriNet.Transition('inv0', label=None)
    inv1 = PetriNet.Transition('inv1', label=None)

    trans = [a, b, c, d, e, f, g, h, inv0, inv1]

    # places
    p0 = PetriNet.Place('p0')
    p1 = PetriNet.Place('p1')
    p2 = PetriNet.Place('p2')
    p3 = PetriNet.Place('p3')
    p4 = PetriNet.Place('p4')
    p5 = PetriNet.Place('p5')
    p6 = PetriNet.Place('p6')
    p7 = PetriNet.Place('p7')
    p8 = PetriNet.Place('p8')
    p9 = PetriNet.Place('p9')
    p10 = PetriNet.Place('p10')
    p11 = PetriNet.Place('p11')

    places = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]

    # arcs
    p0_a = PetriNet.Arc(p0, a)
    p1_c = PetriNet.Arc(p1, c)
    p2_inv0 = PetriNet.Arc(p2, inv0)
    p3_e = PetriNet.Arc(p3, e)
    p4_f = PetriNet.Arc(p4, f)
    p5_g = PetriNet.Arc(p5, g)
    p6_inv1 = PetriNet.Arc(p6, inv1)
    p7_inv1 = PetriNet.Arc(p7, inv1)
    p8_inv1 = PetriNet.Arc(p8, inv1)
    p9_d = PetriNet.Arc(p9, d)
    p10_h = PetriNet.Arc(p10, h)
    p10_b = PetriNet.Arc(p10, b)

    a_p1 = PetriNet.Arc(a, p1)
    c_p2 = PetriNet.Arc(c, p2)
    inv0_p3 = PetriNet.Arc(inv0, p3)
    inv0_p4 = PetriNet.Arc(inv0, p4)
    inv0_p5 = PetriNet.Arc(inv0, p5)
    e_p6 = PetriNet.Arc(e, p6)
    f_p7 = PetriNet.Arc(f, p7)
    g_p8 = PetriNet.Arc(g, p8)
    inv1_p9 = PetriNet.Arc(inv1, p9)
    d_p10 = PetriNet.Arc(d, p10)
    h_p1 = PetriNet.Arc(h, p1)
    b_p11 = PetriNet.Arc(b, p11)

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

    init_marking = Marking([p0])
    final_marking = Marking([p11])

    return net, init_marking, final_marking


def test_marking_equality():
    net = PetriNet()

    p0 = PetriNet.Place('p0')
    p1 = PetriNet.Place('p1')
    net.places.update([p0, p1])

    m0 = Marking([p0])
    m1 = Marking([p0])

    assert m0 == m1


def test_marking_copy():
    net = PetriNet()

    p0 = PetriNet.Place('p0')
    p1 = PetriNet.Place('p1')
    net.places.update([p0, p1])

    m0 = Marking([p0])
    m1 = copy(m0)

    assert m0 == m1


def test_token_replay(ex_petrinet):
    net, init_marking, final_marking = ex_petrinet
    is_inv = lambda t: t.label is None
    place_dict = {p.name:p for p in net.places}

    case = ['a', 'c', 'e', 'g', 'f', 'd', 'b']
    marking_seq = preprocess.get_marking_sequence(case, net, init_marking, final_marking, is_inv)

    expected = [
        ('a', Marking([place_dict['p0']])),
        ('c', Marking([place_dict['p1']])),
        ('e', Marking([place_dict['p2']])),
        ('e', Marking([place_dict['p3'], place_dict['p4'], place_dict['p5']])),
        ('g', Marking([place_dict['p6'], place_dict['p4'], place_dict['p5']])),
        ('f', Marking([place_dict['p6'], place_dict['p4'], place_dict['p8']])),
        ('d', Marking([place_dict['p6'], place_dict['p7'], place_dict['p8']])),
        ('d', Marking([place_dict['p9']])),
        ('b', Marking([place_dict['p10']])),
        (None, Marking([place_dict['p11']])),
    ]

    assert marking_seq == expected

    case_loop = ['a', 'c', 'e', 'g', 'f', 'd', 'h', 'c', 'g', 'f', 'e', 'd', 'b']
    marking_seq = preprocess.get_marking_sequence(case_loop, net, init_marking, final_marking, is_inv)

    expected = [
        ('a', Marking([place_dict['p0']])),
        ('c', Marking([place_dict['p1']])),
        ('e', Marking([place_dict['p2']])),
        ('e', Marking([place_dict['p3'], place_dict['p4'], place_dict['p5']])),
        ('g', Marking([place_dict['p6'], place_dict['p4'], place_dict['p5']])),
        ('f', Marking([place_dict['p6'], place_dict['p4'], place_dict['p8']])),
        ('d', Marking([place_dict['p6'], place_dict['p7'], place_dict['p8']])),
        ('d', Marking([place_dict['p9']])),
        ('h', Marking([place_dict['p10']])),
        ('c', Marking([place_dict['p1']])),
        ('g', Marking([place_dict['p2']])),
        ('g', Marking([place_dict['p3'], place_dict['p4'], place_dict['p5']])),
        ('f', Marking([place_dict['p3'], place_dict['p4'], place_dict['p8']])),
        ('e', Marking([place_dict['p3'], place_dict['p7'], place_dict['p8']])),
        ('d', Marking([place_dict['p6'], place_dict['p7'], place_dict['p8']])),
        ('d', Marking([place_dict['p9']])),
        ('b', Marking([place_dict['p10']])),
        (None, Marking([place_dict['p11']])),
    ]

    assert marking_seq == expected

    case_nonconform = ['a', 'c', 'e']
    marking_seq = preprocess.get_marking_sequence(case_nonconform, net, init_marking, final_marking, is_inv)

    expected = None

    assert marking_seq == expected
