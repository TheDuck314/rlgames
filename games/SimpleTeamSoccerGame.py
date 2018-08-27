# In SimpleTeamSoccerGame the agent moves around like in SlideGame. There is a puck the
# agent can pick up by moving close to it. The agent can shoot the puck, which
# causes the pick to take on a fixed high velocity in the same direction as the
# agent is currently moving. The agent gets reward for shooting the puck into
# a goal region. If it does, the episode ends.

import numpy as np
import copy
import math
from collections import namedtuple

BoundingBox = namedtuple("BoundingBox", ["x1", "y1", "x2", "y2"])

def _inverted_bb(bb):
    return BoundingBox(-bb.x1, bb.y1, -bb.x2, bb.y2)

class KinematicState:
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

    def enforce_bounding_box(self, bb, restitution):
        self.enforce_pos_lower_bound(0, bb.x1, restitution)
        self.enforce_pos_upper_bound(0, bb.x2, restitution)
        self.enforce_pos_lower_bound(1, bb.y1, restitution)
        self.enforce_pos_upper_bound(1, bb.y2, restitution)

    def step(self, accel, friction_coef):
        self.pos += self.vel
        self.vel += (accel - friction_coef * self.vel)

    def inverted(self):
        ret = self.copy()
        ret.pos[0] = -ret.pos[0]
        ret.vel[0] = -ret.vel[0]
        return ret

    def copy(self):
        return KinematicState(
            pos = self.pos.copy(),
            vel = self.vel.copy(),
        )

    def vel_unit_vec(self):
        return self.vel / np.sqrt(np.sum(np.square(self.vel)))

class AgentState:
    def __init__(self, kin, haspuck, bounding_box):
        self.kin = kin
        self.haspuck = haspuck
        self.bounding_box = bounding_box

    def copy(self):
        return AgentState(
            kin = self.kin.copy(),
            haspuck = self.haspuck,
            bounding_box = self.bounding_box
        )

    def inverted(self):
        return AgentState(
            kin = self.kin.inverted(),
            haspuck = self.haspuck,
            bounding_box = _inverted_bb(self.bounding_box),
        )

    def __repr__(self):
        return "Agent(kin={}, haspuck={})".format(self.kin, self.haspuck)

class PuckState:
    def __init__(self, kin, bounding_box):
        self.kin = kin
        self.bounding_box = bounding_box

    def inverted(self):
        return PuckState(
            kin = self.kin.inverted(),
            bounding_box = _inverted_bb(self.bounding_box),
        )

    def copy(self):
        return PuckState(
            kin = self.kin.copy(),
            bounding_box = self.bounding_box
        )

    def __repr__(self):
        return "Puck(kin={})".format(self.kin)

GameState = namedtuple("GameState", ["agents", "puck"])
Action = namedtuple("Action", ["accels", "shoot"])

def int_to_action(choice):
    assert choice in range(18)
    a = choice % 3
    choice /= 3
    b = choice % 3
    choice /= 3
    c = choice
    assert a in [0, 1, 2]
    assert b in [0, 1, 2]
    assert c in [0, 1]
    return Action(accels=np.array([a, b]), shoot=c)

def action_to_int(action):
    return 9*action.shoot + 3*action.accels[1] + action.accels[0]


class DumbAgent:
    def __init__(self):
        pass

    def _choose_action(self, state):
        action = action_to_int(np.random.randint(18))
        value_est = 0.0  #np.random.randn()
        log_p_action = np.log(1/18.0)
        return action, log_p_action, value_est

    def choose_actions(self, states):
        return map(self._choose_action, states)

    def set_be_greedy(self, be_greedy):
        pass


def _sample_from_circle(radius):
    while True:
        x = -1 + 2 * np.random.rand(2)
        if np.sum(np.square(x)) > 1:
            continue  # try again
        x *= radius
        return x

def _sample_from_bb(bb):
    return np.array([bb.x1 + np.random.rand() * (bb.x2 - bb.x1),
                     bb.y1 + np.random.rand() * (bb.y2 - bb.y1)])


class SimpleTeamSoccerGame:
    max_x = 3.0
    max_y = 1.0
    init_agent_vel_radius = 0.1
    init_puck_vel_radius = 0.1

    noise_force_sigma = 0.005
    force = 0.023  # agent acceleration
    agent_friction_coef = 0.25  # friction acceleration = -friction_coef * velocity
    puck_friction_coef = 0.01  # friction acceleration = -friction_coef * velocity
    puck_shoot_speed = 0.35
    puck_acquire_dist = 0.3  # agent picks up pick if it's within this range
    agent_restitution_coef = 0.3
    puck_restitution_coef = 0.9

    win_reward = 1.0
    loss_reward = 0.0  # should anneal this from 0 to -1
    max_frames = 200   # should anneal this from 200 to 2000

    team_size = 2

    def __init__(self):
        agent_states = []
        for i in range(2 * self.team_size):
            is_team0 = i < self.team_size
            bounding_box = BoundingBox(
                x1 = -self.max_x if is_team0 else 0.0,
                y1 = -self.max_y,
                x2 = 0.0 if is_team0 else self.max_x,
                y2 = self.max_y,
            )
            agent_states.append(AgentState(
                kin = KinematicState(
                    pos = _sample_from_bb(bounding_box),
                    vel = _sample_from_circle(self.init_agent_vel_radius),
                ),
                haspuck = False,
                bounding_box = bounding_box,
            ))
        puck_bounding_box = BoundingBox(
            x1 = -np.inf,
            y1 = -self.max_y,
            x2 = np.inf,
            y2 = self.max_y
        )
        puck_state = PuckState(
            kin = KinematicState(
                pos = _sample_from_bb(BoundingBox(-self.max_x, -self.max_y, self.max_x, self.max_y)),
                vel = _sample_from_circle(self.init_puck_vel_radius),
            ),
            bounding_box = puck_bounding_box,
        )
        self.state = GameState(agents=agent_states, puck=puck_state)
        self.frames_left = self.max_frames
        self.finished = False

    @classmethod
    def get_num_agents(cls):
        return 2 * cls.team_size

    def get_observed_states(self):
        # agents always see the state as if they are agent #0 on team0
        team0 = self.state.agents[:self.team_size]
        team1 = self.state.agents[self.team_size:]
        inv_agents = [ag.inverted() for ag in self.state.agents]
        inv_team0 = inv_agents[:self.team_size]
        inv_team1 = inv_agents[self.team_size:]
        inv_puck = self.state.puck.inverted()
        ret = []
        for i in range(self.team_size):  # team0
            ret.append(GameState(
                agents = [team0[i]] + team0[:i] + team0[i+1:] + team1,
                puck = self.state.puck,
            ))
        for i in range(self.team_size):  # team1
            ret.append(GameState(
                agents = [inv_team1[i]] + inv_team1[:i] + inv_team1[i+1:] + inv_team0,
                puck = inv_puck,
            ))
        return ret

    def get_inverted_action(self, action):
        accels = action.accels.copy()
        accels[0] = 2 - accels[0]  # map [0, 1, 2] to [2, 1, 0]
        return Action(accels=accels, shoot=action.shoot)

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
        assert len(actions) == self.get_num_agents()

        #print actions

        # team1 sees an inverted state, so we need to invert their actions
        actions = actions[:self.team_size] + map(self.get_inverted_action, actions[self.team_size:])

        for ac in actions:
            self.validate_action(ac)

        # first do agent kinematics
        new_agents = [agent.copy() for agent in self.state.agents]

        for ac, agent in zip(actions, new_agents):
            # apply accelerations
            # note: accels - 1 maps components in [0, 1, 2] to [-1, 0, 1]
            agent.kin.step(
                accel         = self.force * (ac.accels - 1) + self.gen_noise_force(),
                friction_coef = self.agent_friction_coef)
            # enforce limits of game area
            agent.kin.enforce_bounding_box(agent.bounding_box, self.agent_restitution_coef)


        # quick check that at most one agent should have the puck
        assert sum(ag.haspuck for ag in new_agents) in [0, 1]

        # do puck kinematics and pickup/shoot
        # a shot puck starts from agent's position, traveling in agent's direction,
        # with speed <puck_shoot_speed>
        shot_this_turn = [False] * (2 * self.team_size)
        any_shot_this_turn = False
        for i, (ac, agent) in enumerate(zip(actions, new_agents)):
            if agent.haspuck and ac.shoot == 1:  # agent is shooting
                puck_new_vel = self.puck_shoot_speed * agent.kin.vel_unit_vec()
                new_puck = PuckState(
                    kin = KinematicState(
                        pos = agent.kin.pos + puck_new_vel,
                        vel = puck_new_vel
                    ),
                    bounding_box = self.state.puck.bounding_box,
                )
                new_puck.kin.enforce_bounding_box(new_puck.bounding_box, self.puck_restitution_coef)
                agent.haspuck = False
                shot_this_turn[i] = True
                any_shot_this_turn = True

        if not any(agent.haspuck for agent in new_agents):
            # puck is moving on its own
            # if the puck was shot it already moved for the turn.
            # if it wasn't shot we need to apply its velocity and friction
            if not any_shot_this_turn:
                new_puck = self.state.puck.copy()
                new_puck.kin.step(accel = np.array([0, 0]),
                                  friction_coef = self.puck_friction_coef)
                new_puck.kin.enforce_bounding_box(new_puck.bounding_box, self.puck_restitution_coef)
            # finally check if any agent acquires the puck:
            for i, agent in enumerate(new_agents):
                if shot_this_turn[i]: 
                    continue  # can't pick up puck on turn you shot it
                if np.sign(new_puck.kin.pos[0]) != np.sign(agent.kin.pos[0]):
                    # can only get puck when it's on your side
                    continue  
                if np.sqrt(np.sum(np.square(new_puck.kin.pos - agent.kin.pos))) <= self.puck_acquire_dist:
                    agent.haspuck = True
                    break

        # if an agent has the puck then the puck moves with the agent
        for agent in new_agents:
            if agent.haspuck:
                new_puck = PuckState(
                    kin = agent.kin.copy(),
                    bounding_box = self.state.puck.bounding_box,
                )

        self.state = GameState(agents=new_agents, puck=new_puck)

        if new_puck.kin.pos[0] < -self.max_x:  # team1 wins
            rewards = [self.loss_reward] * self.team_size + [self.win_reward] * self.team_size
            self.finished = True
            #print "right wins!"
        elif new_puck.kin.pos[0] > self.max_x:  # agent2 wins
            rewards = [self.win_reward] * self.team_size + [self.loss_reward] * self.team_size
            self.finished = True
            #print "left wins!"
        else:
            self.frames_left -= 1
            if self.frames_left <= 0:
                self.finished = True
            #    print "game hit frame limit of {}".format(self.max_frames);
            rewards = [0.0] * (2 * self.team_size)

        #if self.finished:
        #    print "loss_reward = {}".format(self.loss_reward)

        return rewards

    def is_finished(self):
        return self.finished

    def get_true_state(self):
        return copy.copy(self.state)



