import base, example_data, utils
import numpy as np


if __name__ == '__main__':
    net1 = example_data.build_net1()
    transcube = example_data.net1_state_trans_matrix()
    emitmat = example_data.net1_emission_matrix()
    confmat = example_data.net1_conformance_matrix()

    n_states = 7
    distmat = np.zeros((n_states, n_states))
    obsmap = {
        'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'g':5
    }
    rev_obsmap = {
        0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'g'
    }
    statemap = dict()

    startprob = [1., 0., 0., 0., 0., 0., 0.]
    startprob = np.reshape(startprob, [1, 7])


    hmm = base.HMMConf(startprob, transcube, emitmat, confmat, distmat, statemap, obsmap, n_states)

    # log move case: <a, b, c, d, e, g> (log move in b or c)
    case = ['a', 'b', 'c', 'd', 'e', 'g']
    case_mapped = list(map(lambda e: obsmap[e], case))
    fwd0 = hmm.forward(case_mapped[0])

    print('log(P(X_1 = <a>, Z_1 = z_1)): {}'.format(fwd0))

    prev_ind = 0
    prev_fwd = fwd0
    case_str = 'a'
    msg0 = '\n   log(P(X_1:{i} = <{case_str}>, Z_{i} = z_{i})): {fwd}'
    msg1 = 'W. P(X_1:{i} = <{case_str}>, Z_{i} = z_{i})): {fwd}'
    msg2 = 'Re-estimated conformance of execution "{}": {}'

    for this_ind in range(1, len(case_mapped)):
        this_act = case_mapped[this_ind]
        prev_act = case_mapped[prev_ind]
        case_str = case_str + ', {}'.format(rev_obsmap[this_act])

        this_fwd = hmm.forward(this_act, prev_act, prev_fwd)
        work_buffer = this_fwd.copy()
        utils.log_normalize(work_buffer, axis=1)
        this_fwd_n = np.exp(work_buffer)
        this_conf = hmm.conform(this_fwd_n, this_act)

        this_msg0 = msg0.format(i=this_ind, case_str=case_str, fwd=this_fwd)
        this_msg1 = msg1.format(i=this_ind, case_str=case_str, fwd=this_fwd_n)
        this_msg2 = msg2.format(rev_obsmap[this_act], this_conf)

        print(this_msg0)
        print(this_msg1)
        print(this_msg2)

        prev_ind = this_ind
        prev_fwd = this_fwd

    print('Finished test')
