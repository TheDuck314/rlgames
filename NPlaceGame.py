# In NPlaceGame there are N discrete locations [0, 1, ..., N-1]. The single
# agent starts on a random location. Each location has a fixed reward which the
# agent receives if it ends a turn on that location. The agent has three
# actions: move one step left, stay put, or move one step right. Attempting to
# move left past 0 or right past N-1 has no effect.

import numpy as np
import copy

class NPlaceGameState:
    """ NPlaceGame's observable state is an int in [0, 1, ..., N-1] giving the
    agent's current location. """
    def __init__(self, location):
        self.location = location

    def __repr__(self):
        return "NPlaceGameState(location={})".format(self.location)


class NPlaceGameAction:
    """ Actions are one of [0, 1, 2] for [left, stay, right] """
    def __init__(self, move):
        assert move in [0, 1, 2]
        self.move = move
 
    def __repr__(self):
        return "NPlaceGameAction(move={})".format(self.move)


class NPlaceGameDumbAgent:
    def __init__(self):
        pass

    def choose_action_and_value_est(self, state):
        move = np.random.choice(3, p=[1/3.0, 1/3.0, 1/3.0])
        action = NPlaceGameAction(move)
        value_est = np.random.randn()
        return action, value_est


class NPlaceGame:
    # no peeking!
    loc_rewards = [-1.0 / 5.0] * 20 + [1.0]
    #end_prob_per_turn = 1.0/200.0
    #end_prob_per_turn = 1.0/5.0
    end_prob_per_turn = 1.0/20.0

    @classmethod
    def get_num_locations(cls):
        return len(cls.loc_rewards)

    @classmethod
    def get_num_actions(cls):
        return 3

    def __init__(self):
        self.state = NPlaceGameState(location=np.random.randint(self.get_num_locations()))
        self.finished = False
        pass

    @classmethod
    def get_num_agents(cls):
        return 1

    def get_observed_states(self):
        return [self.state]

    def simulate_step(self, actions):
        assert not self.finished
        assert len(actions) == 1
        move = actions[0].move
        assert move in [0, 1, 2]

        # update agent's location
        new_loc = self.state.location + (move - 1)
        new_loc = max(0, min(self.get_num_locations() - 1, new_loc))
        self.state = NPlaceGameState(new_loc)

        reward = self.loc_rewards[new_loc]

        if np.random.rand() < self.end_prob_per_turn:
            self.finished = True

        return [reward]

    def is_finished(self):
        return self.finished

    def get_true_state(self):
        return copy.copy(self.state)



