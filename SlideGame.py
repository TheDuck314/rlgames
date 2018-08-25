# In SlideGame the agent moves in R^N and gets reward for being close to a
# destination point. Each step agent applies an acceleration of +1, 0, or -1
# along each of the coordinate axes. Also, there is some friction, and possibly
# a noise acceleration.

import numpy as np
import copy

class SlideGameState:
    """ SlideGame's observable state is the agent's position, the agent's
    velocity, and the destination, which all live in R^N. (Specifically they
    will all be rank-1 np.array's """
    def __init__(self, position, velocity, dest):
        self.position = position
        self.velocity = velocity
        self.dest = dest

    def __repr__(self):
        return "SlideGameState(position={}, velocity={}, dest={})".format(self.position, self.velocity, self.dest)


class SlideGameAction:
    """ Actions are np.array's of three ints, each of which is [-1, 0, or 1] """
    def __init__(self, accels):
        assert isinstance(accels, np.ndarray)
        assert accels.shape == (SlideGame.dim(),)
        assert all(a in [-1, 1] for a in accels)
        self.accels = accels
 
    def __repr__(self):
        return "SlideGameAction(accels={})".format(self.accels)


class SlideGameDumbAgent:
    def __init__(self):
        pass

    def choose_action_and_value_est(self, state):
        accels = np.array([np.random.choice([-1, 1], p=[1/2.0, 1/2.0])
                           for d in range(SlideGame.dim())])
        action = SlideGameAction(accels)
        value_est = np.random.randn()
        return action, value_est


class SlideGame:
    # no peeking!
    init_pos_sigma = 1.0  # radius of normal distribution of initial position
    init_vel_sigma = 0.01  # radius of normal distribution of initial velocities
    dest_pos_sigma = 1.0  # radius of normal distribution of destinations
    noise_force_sigma = 0.001
    force = 0.15  # acceleration per turn
    friction_coef = 0.1  # friction acceleration = -friction_coef * velocity
    end_prob_per_turn = 1.0/20.0
    #end_prob_per_turn = 1.0/200.0
    reward_decay_length = 0.5  # reward per step is exp(-[dist from dest] / reward_decay_length)

    @classmethod
    def dim(cls):
        return 2  # start simple

    def __init__(self):
        self.state = SlideGameState(
            position = self.init_pos_sigma * np.random.randn(self.dim()),
            velocity = self.init_vel_sigma * np.random.randn(self.dim()),
            dest     = self.dest_pos_sigma * np.random.randn(self.dim()),
        )
        self.finished = False
        pass

    def get_num_agents(self):
        return 1

    def get_observed_states(self):
        return [self.state]

    def simulate_step(self, actions):
        assert not self.finished
        assert len(actions) == 1

        # update state
        agent_accel = self.force * actions[0].accels
        friction_accel = -self.friction_coef * self.state.velocity
        noise_accel = self.noise_force_sigma * np.random.randn(self.dim())
        total_accel = agent_accel + friction_accel + noise_accel
        new_pos = self.state.position + self.state.velocity
        new_vel = self.state.velocity + total_accel
        #print "old pos = {}".format(self.state.position)
        #print "old vel = {}".format(self.state.velocity)
        #print "agent_accel = {}".format(agent_accel)
        #print "friction_accel = {}".format(friction_accel)
        #print "noise_accel = {}".format(noise_accel)
        #print "total_accel = {}".format(total_accel)
        #print "new pos = {}".format(new_pos)
        #print "new vel = {}".format(new_vel)

        self.state = SlideGameState(
            position = new_pos,
            velocity = new_vel,
            dest     = self.state.dest,
        )

        dist_to_dest = np.sqrt(np.sum(np.square(self.state.position - self.state.dest)))
        reward = np.exp(-dist_to_dest)

        if np.random.rand() < self.end_prob_per_turn:
            self.finished = True

        return [reward]

    def is_finished(self):
        return self.finished

    def get_replay_state(self):
        return copy.copy(self.state)



