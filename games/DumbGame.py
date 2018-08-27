# DumbGame is a one-player one-step game with N actions. Each action has a
# fixed win probability.

import copy
import numpy as np

class GameState:
    """ DumbGame doesn't really have any state """
    def __init__(self):
        pass

    def __repr__(self):
        return "DumbGame.GameState()"


class Action:
    """ Actions are integers in [0, 1, ..., N-1] """
    def __init__(self, choice):
        assert isinstance(choice, int)
        assert choice >= 0
        self.choice = choice
 
    def __repr__(self):
        return "DumbGame.Action(choice={})".format(self.choice)


class DumbAgent:
    def __init__(self):
        pass

    def _choose_action(self, state):
        choice = 1 if np.random.rand() < 0.5 else 0
        log_p = np.log(0.5)
        action = Action(choice)
        value_est = np.random.randn()
        return action, log_p, value_est

    def choose_actions(self, states):
        return [self._choose_action(st) for st in states]


class DumbGame:
    win_probs = [0.1, 0.5, 0.1, 0.9]  # no peeking!

    def __init__(self):
        self.state = GameState()
        self.finished = False
        pass

    @classmethod
    def get_num_agents(cls):
        return 1

    def get_observed_states(self):
        return [self.state]

    @classmethod
    def get_num_actions(cls):
        return len(cls.win_probs)

    def simulate_step(self, actions):
        assert len(actions) == 1
        # choice i wins with probability win_probs[i]
        choice = actions[0].choice
        assert choice in range(len(self.win_probs))
        win_prob = self.win_probs[choice]
        if np.random.rand() < win_prob:
            reward = +1.0
        else:
            reward = -1.0
        self.finished = True
        return [reward]

    def is_finished(self):
        return self.finished

    def get_true_state(self):
        return copy.copy(self.state)





