# In ShootGame the agent moves around like in SlideGame. There is a puck the
# agent can pick up by moving close to it. The agent can shoot the puck, which
# causes the pick to take on a fixed high velocity in the same direction as the
# agent is currently moving. The agent gets reward for shooting the puck into
# a goal region. If it does, the episode ends.

import numpy as np
import copy
import math
from collections import namedtuple

ShootGameObjState = namedtuple("ShootGameObjState", ["position", "velocity"])
ShootGameState = namedtuple("ShootGameState", ["agent", "puck", "haspuck"])
ShootGameAction = namedtuple("ShootGameAction", ["accels", "shoot"])

class ShootGameDumbAgent:
    def __init__(self):
        pass

    def choose_action_and_value_est(self, state):
        accels = np.array([np.random.choice([-1, 0, 1], p=[1/3.0, 1/3.0,  1/3.0])
                           for d in range(ShootGame.dim)])
        shoot = np.random.rand() < 0.1
        action = ShootGameAction(accels=accels, shoot=shoot)
        value_est = np.random.randn()
        return action, value_est

def sample_from_ball(N, radius):
    while True:
        x = -1 + 2 * np.random.rand(N)
        if np.sum(np.square(x)) > 1:
            continue  # try again
        x *= radius
        return x


class ShootGame:
    # no peeking!
    init_agent_pos_radius = 2.0  # radius of distribution of initial position
    init_agent_vel_radius = 0.01  # radius of distribution of initial velocities
    init_puck_pos_radius = 2.0  # radius of distribution of initial position
    init_puck_vel_radius = 0.01  # radius of distribution of initial velocities
    goal_radius = 3.0
    noise_force_sigma = 0.001
    force = 0.15  # agent acceleration
    agent_friction_coef = 0.5  # friction acceleration = -friction_coef * velocity
    puck_friction_coef = 0.1  # friction acceleration = -friction_coef * velocity
    end_prob_per_turn = 1.0/200.0
    puck_shoot_speed = 1.0
    puck_acquire_dist = 0.3  # agent picks up pick if it's within this range
    max_frames = 100
    #goal_opening_half_angle = math.pi
    goal_opening_half_angle = 0.15 * math.pi / 2

    dim = 2

    def __init__(self):
        self.state = ShootGameState(
            agent = ShootGameObjState(position = sample_from_ball(self.dim, self.init_agent_pos_radius),
                                      velocity = sample_from_ball(self.dim, self.init_agent_vel_radius)),
            puck = ShootGameObjState(position = sample_from_ball(self.dim, self.init_puck_pos_radius),
                                     velocity = sample_from_ball(self.dim, self.init_puck_vel_radius)),
            haspuck = False,
        )
        self.frames_left = self.max_frames
        self.finished = False
        pass

    def get_num_agents(self):
        return 1

    def get_observed_states(self):
        return [self.state]

    def validate_action(self, action):
        assert isinstance(action.accels, np.ndarray)
        assert action.accels.shape == (self.dim,)
        assert all(a in [-1, 0, 1] for a in action.accels)
        assert action.shoot in [True, False]

    def simulate_step(self, actions):
        assert not self.finished
        assert len(actions) == 1
        self.validate_action(actions[0])

        #print "old agent pos = {}".format(self.state.agent.position)
        #print "old agent vel = {}".format(self.state.agent.velocity)
        #print "old puck pos = {}".format(self.state.puck.position)
        #print "old puck vel = {}".format(self.state.puck.velocity)
        #print "old haspuck = {}".format(self.state.haspuck)

        # update state
        # agent
        agent_action_accel   = self.force * actions[0].accels
        agent_friction_accel = -self.agent_friction_coef * self.state.agent.velocity
        agent_noise_accel    = self.noise_force_sigma * np.random.randn(self.dim)
        agent_total_accel = agent_action_accel + agent_friction_accel + agent_noise_accel
        agent_new_pos = self.state.agent.position + self.state.agent.velocity
        agent_new_vel = self.state.agent.velocity + agent_total_accel
        #print "agent_action_accel = {}".format(agent_action_accel)
        #print "agent_friction_accel = {}".format(agent_friction_accel)
        #print "agent_noise_accel = {}".format(agent_noise_accel)
        #print "agent_total_accel = {}".format(agent_total_accel)
        #print "agent_new_pos = {}".format(agent_new_pos)
        #print "agent_new_vel = {}".format(agent_new_vel)
        new_haspuck = self.state.haspuck
        if self.state.haspuck:
            if actions[0].shoot:
                # a shot puck starts from agent's position, traveling in agent's direction,
                # with speed <puck_shoot_speed>
                #print "shooting puck!"
                agent_vel_unit_vec = agent_new_vel / np.sqrt(np.sum(np.square(agent_new_vel)))
                puck_new_vel = self.puck_shoot_speed * agent_vel_unit_vec
                puck_new_pos = agent_new_pos + puck_new_vel
                new_haspuck = False
        else:
            # puck is moving on its own
            #print "puck is moving on its own"
            puck_friction_accel = -self.puck_friction_coef * self.state.puck.velocity
            puck_new_pos = self.state.puck.position + self.state.puck.velocity
            puck_new_vel = self.state.puck.velocity + puck_friction_accel
            #print "puck_friction_accel = {}".format(puck_friction_accel)
            #print "puck_new_pos = {}".format(puck_new_pos)
            #print "puck_new_vel = {}".format(puck_new_vel)
            puck_dist = np.sqrt(np.sum(np.square(puck_new_pos - agent_new_pos)))
            if puck_dist <= self.puck_acquire_dist:
                #print "picking up puck!"
                new_haspuck = True  # agent picks up puck
        if new_haspuck:
            # puck travels with agent
            #print "puck travelling with agent"
            puck_new_pos = agent_new_pos
            puck_new_vel = agent_new_vel
        #print "final new_haspuck = {}".format(new_haspuck)
        #print "final puck_new_pos = {}".format(puck_new_pos)
        #print "final puck_new_vel = {}".format(puck_new_vel)
        self.state = ShootGameState(
            agent = ShootGameObjState(position = agent_new_pos,
                                      velocity = agent_new_vel),
            puck = ShootGameObjState(position = puck_new_pos,
                                     velocity = puck_new_vel),
            haspuck = new_haspuck,
        )

        self.frames_left -= 1

        puck_dist_from_origin = np.sqrt(np.sum(np.square(self.state.puck.position)))
        if puck_dist_from_origin >= self.goal_radius:
            # we only win if the puck is in a certain angle
            puck_angle = math.atan2(self.state.puck.position[1], self.state.puck.position[0])
            if abs(puck_angle) < self.goal_opening_half_angle: 
                print "puck pos is {}, puck_angle is {}, WIN!".format(self.state.puck.position, puck_angle)
                reward = 1.0
            else:
                reward = 0.0
                print "puck pos is {}, puck_angle is {}, LOSS!".format(self.state.puck.position, puck_angle)
            self.finished = True
        elif self.frames_left <= 0:
            reward = 0.0
            self.finished = True
        else:
            reward = 0.0  # todo

        return [reward]

    def is_finished(self):
        return self.finished

    def get_true_state(self):
        return copy.copy(self.state)



