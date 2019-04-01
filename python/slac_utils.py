import numpy as np
import pandas as pd
import utils, warnings


class HMMConf(Logged):
    def __init__(self, startprob, trans_cube, emit_mat, conf_mat, 
                 state_map, activity_map, *args, **kwargs):
        super().__init__(*args, **kwargs)

        msg = """Number of {item} are different between state transition cube and emission matrix

        [trans_cube]: {left}
        [emit_mat]: {right}
        """

        act_msg = msg.format(item='activities', left=trans_cube.shape[0], right=emit_mat.shape[1])
        state_msg = msg.format(item='states', left=trans_cube.shape[1], right=emit_mat.shape[0])
        assert trans_cube.shape[0] == emit_mat.shape[1], act_msg
        assert trans_cube.shape[1] == emit_mat.shape[0], state_msg

        self.n_states = emit_mat.shape[0]
        self.startprob = startprob
        self.trans_cube = trans_cube
        self.emit_mat = emit_mat
        self.conf_mat = conf_mat
        self.trans_cube_d = np.ones(trans_cube.shape) / trans_cube.shape[1]
        self.emit_mat_d = np.ones(emit_mat.shape) / emit_mat.shape[1]
        self.state_map = state_map
        self.activity_map = activity_map

    def conform(self, state_est, act):
        """
        Computes conformance given a state estimation and observed activity.

        :param state_est: state estimation at time t
        :param act: activity 
        """
        if (not np.isclose(state_est.sum(), [1.])):
            raise ValueError('State estimation: {} does not sum to 1.'.format(state_est))

        return np.dot(state_est, self.conf_mat[act])

    def emissionprob(self, state_est, act):
        """
        Computes P(x is act at time t | z at time t) where x is the observation variable
        and z is the state variable. 

        :param state_est: state estimation at time t
        :param act: observed activity at time t
        """
        if (state_est.shape[1] != self.emit_mat.shape[0]):
            raise ValueError('Invalid state length: {}'.format(state_est))

        obs_likelihood = np.sum(state_est)
        conf = [0.]
        if obs_likelihood > 0:
            state_conditioned = state_est / obs_likelihood
            conf = self.conform(state_conditioned, act)
        else:
            msg = 'State estimation sums to 0. This makes conformance equal to 0.'
            warnings.warn(msg, category=UserWarning)

        msg = 'Conformance between state and observation at time t' \
              'before observation adjustment: {:.2f}'.format(conf[0]))
        self.logger.info(msg)

        return conf * self.emit_mat[:,act] + (1 - conf) * self.emit_mat_d[:,act]

    def stateprob(self, state_est, act):
        """
        Computes P(z at time t + 1 | z at time t, x is act at time t) where x is the observation
        variable and z is the state variable.

        :param state_est: state estimation at time t
        :param act: observed activity at time t + 1
        """
        if (state_est.shape[1] != self.stateprob[0,:,:].shape[1]):
            raise ValueError('Invalid state length: {}'.format(state_est))

        obs_likelihood = np.sum(state_est)
        conf = [0.]
        if obs_likelihood > 0:
            state_conditioned = state_est / obs_likelihood
            conf = self.conform(state_conditioned, act)
        else:
            msg = 'State estimation sums to 0. This makes conformance equal to 0.'
            warnings.warn(msg, category=UserWarning)

        msg = 'Conformance between state and observation at time t' \
              'before observation adjustment: {:.2f}'.format(conf[0]))
        self.logger.info(msg)

        return conf * self.trans_cube[act,:,:] + (1 - conf) * self.trans_cube_d[act,:,:]

    def forward(act, prev_act, prev_frwd=None, init=None):
        """
        :param act: observed activity at time t
        :param prev_act: observed activity at time t - 1
        :param prev_frwd: forward probability at time t
        :param init: initial probability 
        """
        if prev_frwd is None:
            return init * self.emissionprob(init, act)

        est_cur_state = (self.stateprob(prev_frwd, prev_act).T * prev_frwd).sum(axis=1)
        est_cur_state = est_cur_state.reshape([1, self.trans_cube[0,:,:].shape[1]])

        msg = '   State estimate of time t before observation at time t: {}'.format(est_cur_state)
        self.logger.info(msg)
        msg = 'W. State estimate of time t before observation at time t: {}'.format(est_cur_state / est_cur_state.sum())

        obs_likelihood = self.emissionprob(est_cur_state, act)
        msg = 'Likelihood of observation at states time t: {}'.format(obs_likelihood)
        self.logger.info(msg)
        return obs_likelihood * est_cur_state
