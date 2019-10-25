import numpy as np
import pytest
from pm4py.objects.transition_system import transition_system as ts
from pm4py.objects.transition_system import utils


from hmmconf import hmmconf_setup


class TestComputeDistmat:
    def test_simple_case(self):
        g = ts.TransitionSystem()
        m0 = ts.TransitionSystem.State('m0')
        m1 = ts.TransitionSystem.State('m1')
        m2 = ts.TransitionSystem.State('m2')

        t1 = ts.TransitionSystem.Transition('t1', m0, m1)
        t2 = ts.TransitionSystem.Transition('t2', m1, m2)

        m0.outgoing.add(t1)
        m1.outgoing.add(t2)
        m1.incoming.add(t1)
        m2.incoming.add(t2)

        g.states.add(m0)
        g.states.add(m1)
        g.states.add(m2)
        g.transitions.add(t1)
        g.transitions.add(t2)

        is_inv = lambda t: t.name is None
        state2int = {
            'm0': 0,
            'm1': 1,
            'm2': 2
        }
        distmat = hmmconf_setup.compute_distmat(g, state2int, is_inv)

        expected = np.zeros((3, 3))
        expected[0,1] = 1
        expected[0,2] = 2
        expected[1,0] = 1
        expected[1,2] = 1
        expected[2,0] = 2
        expected[2,1] = 1

        assert distmat.shape == expected.shape
        assert (distmat == expected).all()

    def test_simple_inv_case(self):
        g = ts.TransitionSystem()
        m0 = ts.TransitionSystem.State('m0')
        m1 = ts.TransitionSystem.State('m1')
        m2 = ts.TransitionSystem.State('m2')

        inv = ts.TransitionSystem.Transition(None, m0, m1)
        t2 = ts.TransitionSystem.Transition('t2', m1, m2)
        t3 = ts.TransitionSystem.Transition('t3', m0, m2)
        m0.outgoing.add(inv)
        m0.outgoing.add(t3)
        m1.outgoing.add(t2)
        m1.incoming.add(inv)
        m2.incoming.add(t2)
        m2.incoming.add(t3)

        g.states.add(m0)
        g.states.add(m1)
        g.states.add(m2)
        g.transitions.add(inv)
        g.transitions.add(t2)
        g.transitions.add(t3)

        is_inv = lambda t: t.name is None
        state2int = {
            'm0': 0,
            'm1': 1,
            'm2': 2
        }
        distmat = hmmconf_setup.compute_distmat(g, state2int, is_inv)

        expected = np.zeros((3, 3))
        expected[0,1] = 0
        expected[0,2] = 1
        expected[1,0] = 0
        expected[1,2] = 1
        expected[2,0] = 1
        expected[2,1] = 1

        assert distmat.shape == expected.shape
        assert (distmat == expected).all()

    def test_inv_loop_case(self):
        g = ts.TransitionSystem()
        m0 = ts.TransitionSystem.State('m0')
        m1 = ts.TransitionSystem.State('m1')
        m2 = ts.TransitionSystem.State('m2')
        m3 = ts.TransitionSystem.State('m3')

        inv0 = ts.TransitionSystem.Transition(None, m0, m1)
        inv1 = ts.TransitionSystem.Transition(None, m1, m2)
        inv2 = ts.TransitionSystem.Transition(None, m2, m3)
        t3 = ts.TransitionSystem.Transition(None, m0, m3)

        m0.outgoing.add(inv0)
        m1.outgoing.add(inv1)
        m2.outgoing.add(inv2)
        m0.outgoing.add(t3)
        m1.incoming.add(inv0)
        m2.incoming.add(inv1)
        m3.incoming.add(inv2)
        m3.incoming.add(t3)

        g.states.add(m0)
        g.states.add(m1)
        g.states.add(m2)
        g.states.add(m3)
        g.transitions.add(inv0)
        g.transitions.add(inv1)
        g.transitions.add(inv2)
        g.transitions.add(t3)

        is_inv = lambda t: t.name is None
        state2int = {
            'm0': 0,
            'm1': 1,
            'm2': 2,
            'm3': 3,
        }
        distmat = hmmconf_setup.compute_distmat(g, state2int, is_inv)

        expected = np.zeros((4, 4))

        assert distmat.shape == expected.shape
        assert (distmat == expected).all()


class TestComputeConfmat:
    def test_simple_case(self):
        g = ts.TransitionSystem()

        m0 = ts.TransitionSystem.State('m0')
        m1 = ts.TransitionSystem.State('m1')
        m2 = ts.TransitionSystem.State('m2')
        m3 = ts.TransitionSystem.State('m3')

        t1 = ts.TransitionSystem.Transition('t1', m0, m1)
        t2 = ts.TransitionSystem.Transition('t2', m0, m2)
        t3 = ts.TransitionSystem.Transition('t3', m1, m3)
        m0.outgoing.add(t1)
        m1.incoming.add(t1)
        m0.outgoing.add(t2)
        m2.incoming.add(t2)
        m1.outgoing.add(t3)
        m3.incoming.add(t3)

        g.states.add(m0)
        g.states.add(m1)
        g.states.add(m2)
        g.states.add(m3)
        g.transitions.add(t1)
        g.transitions.add(t2)
        g.transitions.add(t3)

        is_inv = lambda t: t.name is None
        state2int = {
            m0.name: 0,
            m1.name: 1,
            m2.name: 2,
            m3.name: 3
        }
        obs2int = {
            t1.name: 0,
            t2.name: 1,
            t3.name: 2
        }

        confmat = hmmconf_setup.compute_confmat(g, m0, is_inv, state2int, obs2int)

        expected = np.zeros((3, 4))
        expected[0][0] = 1
        expected[1][0] = 1
        expected[2][1] = 1

        print(confmat)

        assert confmat.shape == expected.shape
        assert (confmat == expected).all()

    def test_xor_with_inv_case(self):
        g = ts.TransitionSystem()
        
        m0 = ts.TransitionSystem.State('m0')
        m1 = ts.TransitionSystem.State('m1')
        m2 = ts.TransitionSystem.State('m2')

        inv = ts.TransitionSystem.Transition(None, m0, m1)
        t1 = ts.TransitionSystem.Transition('t1', m0, m1)
        t2 = ts.TransitionSystem.Transition('t2', m1, m2)
        m0.outgoing.add(inv)
        m0.outgoing.add(t1)
        m1.outgoing.add(t2)
        m1.incoming.add(inv)
        m1.incoming.add(t1)
        m2.incoming.add(t2)

        g.states.add(m0)
        g.states.add(m1)
        g.states.add(m2)
        g.transitions.add(inv)
        g.transitions.add(t1)
        g.transitions.add(t2)

        is_inv = lambda t: t.name is None
        state2int = {
            m0.name: 0,
            m1.name: 1,
            m2.name: 2
        }
        obs2int = {
            t1.name: 0,
            t2.name: 1
        }

        confmat = hmmconf_setup.compute_confmat(g, m0, is_inv, state2int, obs2int)
        expected = np.zeros((2, 3))
        expected[0][0] = 1
        expected[1][0] = 1
        expected[1][1] = 1

        assert confmat.shape == expected.shape
        assert (confmat == expected).all()

    def test_fwdprop_inv_case(self):
        g = ts.TransitionSystem()
        
        m0 = ts.TransitionSystem.State('m0')
        m1 = ts.TransitionSystem.State('m1')
        m2 = ts.TransitionSystem.State('m2')
        m3 = ts.TransitionSystem.State('m3')

        t1 = ts.TransitionSystem.Transition('t1', m0, m1)
        inv0 = ts.TransitionSystem.Transition(None, m1, m2)
        inv1 = ts.TransitionSystem.Transition(None, m2, m3)
        m0.outgoing.add(t1)
        m1.outgoing.add(inv0)
        m2.outgoing.add(inv1)
        m1.incoming.add(t1)
        m2.incoming.add(inv0)
        m3.incoming.add(inv1)

        g.states.add(m0)
        g.states.add(m1)
        g.states.add(m2)
        g.states.add(m3)
        g.transitions.add(t1)
        g.transitions.add(inv0)
        g.transitions.add(inv1)

        is_inv = lambda t: t.name is None
        state2int = {
            m0.name: 0,
            m1.name: 1,
            m2.name: 2,
            m3.name: 3,
        }
        obs2int = {
            t1.name: 0,
        }

        confmat = hmmconf_setup.compute_confmat(g, m0, is_inv, state2int, obs2int)
        expected = np.zeros((1, 4))
        expected[0][0] = 1

        assert confmat.shape == expected.shape
        assert (confmat == expected).all()

    def test_multixor_inv_case(self):
        g = ts.TransitionSystem()

        m0 = ts.TransitionSystem.State('m0')
        m1 = ts.TransitionSystem.State('m1')
        m2 = ts.TransitionSystem.State('m2')
        m3 = ts.TransitionSystem.State('m3')
        m4 = ts.TransitionSystem.State('m4')
        m5 = ts.TransitionSystem.State('m5')

        t3 = ts.TransitionSystem.Transition('t3', m2, m3)
        t4 = ts.TransitionSystem.Transition('t4', m0, m4)
        t5 = ts.TransitionSystem.Transition('t5', m1, m5)
        inv0 = ts.TransitionSystem.Transition(None, m0, m1)
        inv1 = ts.TransitionSystem.Transition(None, m1, m2)

        m0.outgoing.add(inv0)
        m1.incoming.add(inv0)
        m0.outgoing.add(t4)
        m4.incoming.add(t4)
        m1.outgoing.add(inv1)
        m2.incoming.add(inv1)
        m1.outgoing.add(t5)
        m5.incoming.add(t5)
        m2.outgoing.add(t3)
        m3.incoming.add(t3)

        is_inv = lambda t: t.name is None
        state2int = {
            m0.name: 0,
            m1.name: 1,
            m2.name: 2,
            m3.name: 3,
            m4.name: 4,
            m5.name: 5,
        }
        obs2int = {
            t3.name: 0,
            t4.name: 1,
            t5.name: 2,
        }

        confmat = hmmconf_setup.compute_confmat(g, m0, is_inv, state2int, obs2int)
        expected = np.zeros((3, 6))
        expected[0][0] = 1
        expected[1][0] = 1
        expected[2][0] = 1
        expected[0][1] = 1
        expected[2][1] = 1
        expected[0][2] = 1

        assert confmat.shape == expected.shape
        assert (confmat == expected).all()
