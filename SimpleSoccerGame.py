# In SimpleSoccerGame the agent moves around like in SlideGame. There is a puck the
# agent can pick up by moving close to it. The agent can shoot the puck, which
# causes the pick to take on a fixed high velocity in the same direction as the
# agent is currently moving. The agent gets reward for shooting the puck into
# a goal region. If it does, the episode ends.

import numpy as np
import copy
import math
from collections import namedtuple

class SimpleSoccerKinematicState:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel

    def __repr__(self):
        return "Kin(pos=({},{}), vel=({},{}))".format(self.pos[0], self.pos[1], self.vel[0], self.vel[1])

    def enforce_pos_lower_bound(self, d, coord, restitution):
        if self.pos[d] < coord:
            self.pos[d] += coord - self.pos[d]
            if self.vel[d] < 0:
                self.vel[d] *= -restitution
    
    def enforce_pos_upper_bound(self, d, coord, restitution):
        if self.pos[d] > coord:
            self.pos[d] -= self.pos[d] - coord
            if self.vel[d] > 0:
                self.vel[d] *= -restitution

    def enforce_bounding_box(self, x1, y1, x2, y2, restitution):
        self.enforce_pos_lower_bound(0, x1, restitution)
        self.enforce_pos_upper_bound(0, x2, restitution)
        self.enforce_pos_lower_bound(1, y1, restitution)
        self.enforce_pos_upper_bound(1, y2, restitution)

    def step(self, accel, friction_coef):
        self.pos += self.vel
        self.vel += (accel - friction_coef * self.vel)

    def inverted(self):
        ret = self.copy()
        ret.pos[0] = -ret.pos[0]
        ret.vel[0] = -ret.vel[0]
        return ret

    def copy(self):
        return SimpleSoccerKinematicState(
            pos = self.pos.copy(),
            vel = self.vel.copy(),
        )


SimpleSoccerState = namedtuple("SimpleSoccerState", ["agent0", "agent1", "puck", "haspuck0", "haspuck1"])
SimpleSoccerAction = namedtuple("SimpleSoccerAction", ["accels", "shoot"])

def make_SimpleSoccerAction(choice):
    assert choice in range(18)
    a = choice % 3
    choice /= 3
    b = choice % 3
    choice /= 3
    c = choice
    assert a in [0, 1, 2]
    assert b in [0, 1, 2]
    assert c in [0, 1]
    return SimpleSoccerAction(accels=np.array([a, b]), shoot=c)

def SimpleSoccerAction_toint(action):
    return 9*action.shoot + 3*action.accels[1] + action.accels[0]


class SimpleSoccerDumbAgent:
    def __init__(self):
        pass

    def choose_action(self, state):
        accels = np.array([np.random.choice(3, p=[1/3.0, 1/3.0,  1/3.0])
                           for d in range(2)])
        shoot = 1 if np.random.rand() < 0.1 else 0
        action = SimpleSoccerAction(accels=accels, shoot=shoot)
        value_est = 0.0  #np.random.randn()
        log_p_action = -0.5  # whatever
        return action, log_p_action, value_est

    def set_be_greedy(self, be_greedy):
        pass


def _sample_from_circle(radius):
    while True:
        x = -1 + 2 * np.random.rand(2)
        if np.sum(np.square(x)) > 1:
            continue  # try again
        x *= radius
        return x

def _sample_from_rect(x1, y1, x2, y2):
    return np.array([x1 + np.random.rand() * (x2 - x1),
                     y1 + np.random.rand() * (y2 - y1)])


class SimpleSoccerGame:
    max_x = 3.0
    max_y = 1.0
    init_agent_vel_radius = 0.1
    init_puck_vel_radius = 0.1

    noise_force_sigma = 0.005
    force = 0.023  # agent acceleration
    agent_friction_coef = 0.25  # friction acceleration = -friction_coef * velocity
    puck_friction_coef = 0.01  # friction acceleration = -friction_coef * velocity
    #puck_shoot_speed = 0.35
    #puck_shoot_speed = 0.43
    #puck_shoot_speed = 0.55
    puck_shoot_speed = 0.35
    puck_acquire_dist = 0.3  # agent picks up pick if it's within this range
    agent_restitution_coef = 0.3
    puck_restitution_coef = 0.9

    win_reward = 1.0
    loss_reward = 0.0  # should anneal this from 0 to -1
    max_frames = 200   # should anneal this from 100 to 1000

    def __init__(self):
        self.state = SimpleSoccerState(
            agent0 = SimpleSoccerKinematicState(
                pos = _sample_from_rect(-self.max_x, -self.max_y, 0.0, self.max_y),
                vel = _sample_from_circle(self.init_agent_vel_radius)),
            agent1 = SimpleSoccerKinematicState(
                pos = _sample_from_rect(0.0, -self.max_y, self.max_x, self.max_y),
                vel = _sample_from_circle(self.init_agent_vel_radius)),
            puck = SimpleSoccerKinematicState(
                pos = _sample_from_rect(-self.max_x, -self.max_y, self.max_x, self.max_y),
                #pos = _sample_from_rect(-self.max_x, -self.max_y, 0.0, self.max_y),
                vel = _sample_from_circle(self.init_puck_vel_radius)),
            haspuck0 = False,
            haspuck1 = False,
        )
        self.frames_left = self.max_frames
        self.finished = False

    @classmethod
    def get_num_agents(cls):
        return 2

    def get_inverted_state(self):
        return SimpleSoccerState(
            agent0   = self.state.agent1.inverted(),
            agent1   = self.state.agent0.inverted(),
            puck     = self.state.puck.inverted(),
            haspuck0 = self.state.haspuck1,
            haspuck1 = self.state.haspuck0,
        )

    def get_inverted_action(self, action):
        accels = action.accels.copy()
        accels[0] = 2 - accels[0]  # map [0, 1, 2] to [2, 1, 0]
        return SimpleSoccerAction(accels=accels, shoot=action.shoot)

    def get_observed_states(self):
        return [self.state, self.get_inverted_state()]

    def validate_action(self, action):
        assert isinstance(action.accels, np.ndarray)
        assert action.accels.shape == (2,)
        assert all(a in [0, 1, 2] for a in action.accels)
        assert action.shoot in [0, 1]

    def gen_noise_force(self):
        return self.noise_force_sigma * np.random.randn(2)

    def simulate_step(self, actions):
        #print "simulate_step frames_left = {}".format(self.frames_left)
        assert not self.finished
        assert len(actions) == 2

        #print actions

        # agent1 sees an inverted state, so we need to invert its action
        actions = [actions[0], self.get_inverted_action(actions[1])]

        for action in actions:
            self.validate_action(action)

        # first do agent kinematics
        new_agent0 = self.state.agent0.copy()
        new_agent1 = self.state.agent1.copy()

        # note: accels - 1 maps components in [0, 1, 2] to [-1, 0, 1]
        new_agent0.step(
            accel         = self.force * (actions[0].accels - 1) + self.gen_noise_force(),
            friction_coef = self.agent_friction_coef)
        new_agent1.step(
            accel         = self.force * (actions[1].accels - 1) + self.gen_noise_force(),
            friction_coef = self.agent_friction_coef)

        new_agent0.enforce_bounding_box(-self.max_x,
                                        -self.max_y,
                                        0,
                                        self.max_y,
                                        self.agent_restitution_coef)
        new_agent1.enforce_bounding_box(0,
                                        -self.max_y,
                                        self.max_x,
                                        self.max_y,
                                        self.agent_restitution_coef)


        new_haspuck0 = self.state.haspuck0
        new_haspuck1 = self.state.haspuck1
        assert not (new_haspuck0 and new_haspuck1)

        # next do puck kinematics and pickup/shoot
        # a shot puck starts from agent's position, traveling in agent's direction,
        # with speed <puck_shoot_speed>
        shot_this_turn0 = False
        shot_this_turn1 = False
        if new_haspuck0:
            if actions[0].shoot == 1:
                #print "agent 0 shooting!"
                puck_new_vel = self.puck_shoot_speed * new_agent0.vel / np.sqrt(np.sum(np.square(new_agent0.vel)))
                new_puck = SimpleSoccerKinematicState(
                    pos = new_agent0.pos + puck_new_vel,
                    vel = puck_new_vel)
                new_haspuck0 = False
                shot_this_turn0 = True
        elif new_haspuck1:
            if actions[1].shoot == 1:
                #print "agent 1 shooting!"
                puck_new_vel = self.puck_shoot_speed * new_agent1.vel / np.sqrt(np.sum(np.square(new_agent1.vel)))
                new_puck = SimpleSoccerKinematicState(
                    pos = new_agent1.pos + puck_new_vel,
                    vel = puck_new_vel)
                new_haspuck1 = False
                shot_this_turn1 = True

        if (not new_haspuck0) and (not new_haspuck1):
            # puck is moving on its own
            # if the puck was shot it already moved for the turn.
            # if it wasn't we need to apply its velocity and friction
            if (not shot_this_turn0) and (not shot_this_turn1):
                new_puck = self.state.puck.copy()
                new_puck.step(accel = np.array([0, 0]),
                              friction_coef = self.puck_friction_coef)
            # finally check if either agent acquires puck:
            if new_puck.pos[0] < 0 and not shot_this_turn0:
                if np.sqrt(np.sum(np.square(new_puck.pos - new_agent0.pos))) <= self.puck_acquire_dist:
                    new_haspuck0 = True
            elif new_puck.pos[0] > 0 and not shot_this_turn1:
                if np.sqrt(np.sum(np.square(new_puck.pos - new_agent1.pos))) <= self.puck_acquire_dist:
                    new_haspuck1 = True

        # if an agent has the puck then the puck moves with the agent
        if new_haspuck0:
            new_puck = new_agent0.copy()
        elif new_haspuck1:
            new_puck = new_agent1.copy()
        else:
            new_puck.enforce_bounding_box(
                -np.inf,
#                -self.max_x,
                -self.max_y,
                np.inf,
                self.max_y,
                self.puck_restitution_coef,
            )

        self.state = SimpleSoccerState(
            agent0   = new_agent0,
            agent1   = new_agent1,
            puck     = new_puck,
            haspuck0 = new_haspuck0,
            haspuck1 = new_haspuck1,
        )

        if new_puck.pos[0] < -self.max_x:  # agent1 wins
            rewards = [self.loss_reward, self.win_reward]
            self.finished = True
            #print "right wins!"
        elif new_puck.pos[0] > self.max_x:  # agent2 wins
            rewards = [self.win_reward, self.loss_reward]
            self.finished = True
            #print "left wins!"
        else:
            self.frames_left -= 1
            if self.frames_left <= 0:
                self.finished = True
            #    print "game hit frame limit of {}".format(self.max_frames);
            rewards = [0.0, 0.0]

        #if self.finished:
        #    print "loss_reward = {}".format(self.loss_reward)

        return rewards

    def is_finished(self):
        return self.finished

    def get_true_state(self):
        return copy.copy(self.state)



