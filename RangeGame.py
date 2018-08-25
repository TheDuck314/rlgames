# RangeGame is a one-step game. The state is an observable random vector in
# R^S. There are N actions. Each action has an associated hyper-rectangle in
# R^S. The winning action is the first action such that the state vector falls
# in that action's hyper-rectangle. The last action always has range equal to 
# all of R^S

import numpy as np

class RangeGameState:
    """ RangeGame's state is a 1-d array of floats. """

    @staticmethod
    def dim():
        return 1
        #return 2

    def __init__(self, number):
        self.numbers = np.random.randn(self.dim())

    def __repr__(self):
        return "RangeGameState(numbers={})".format(self.numbers)


class RangeGameAction:
    """ Actions are numbers in [0, 1, ..., N-1] """
    def __init__(self, choice):
        assert isinstance(choice, int)
        assert choice >= 0
        self.choice = choice
 
    def __repr__(self):
        return "RangeGameAction(choice={})".format(self.choice)


class RangeGameDumbAgent:
    def __init__(self):
        pass

    def choose_action_and_value_est(self, state):
        choice = 1 if np.random.rand() < 0.5 else 0
        action = RangeGameAction(choice)
        value_est = np.random.randn()
        return action, value_est


class RangeGame:
    # no peeking!
    action_hyperrects = [
        [[   -0.5,    0.5]],
        [[-np.inf, np.inf]],
    ]
#    action_hyperrects = [
#        [[ -np.inf,    0.0]],
#        [[     0.0, np.inf]],
#    ]
#    action_hyperrects = [
#        [[ -np.inf,    0.0], [-np.inf, np.inf]],
#        [[     0.0, np.inf], [-np.inf, np.inf]],
#    ]
#    action_hyperrects = [
#        [[ 0.0, np.inf], [ 0.0, np.inf]],
#        [[-np.inf, 0.0], [-np.inf, 0.0]],
#        [[-np.inf, np.inf], [-np.inf, np.inf]],
#    ]
#    action_hyperrects = [
#        [[ 0.0, 1.0], [ 0.0, 1.0]],
#        [[-1.0, 0.0], [-1.0, 0.0]],
#        [[-np.inf, np.inf], [-np.inf, np.inf]],
#    ]

    def __init__(self):
        self.state = RangeGameState(number=np.random.randn())
        self.finished = False
        pass

    def get_num_agents(self):
        return 1

    def get_observed_states(self):
        return [self.state]

    @classmethod
    def get_num_actions(cls):
        return len(cls.action_hyperrects)

    @staticmethod
    def state_in_hyperrect(state, hyperrect):
        assert len(state.numbers) == RangeGameState.dim()
        assert len(hyperrect) == RangeGameState.dim()
        for d in range(RangeGameState.dim()):
            if not (hyperrect[d][0] <= state.numbers[d] < hyperrect[d][1]):
                return False
        return True

    def simulate_step(self, actions):
        assert len(actions) == 1
        choice = actions[0].choice
        assert choice in range(self.get_num_actions())

        winning_choice = -1
        for i, hyperrect in enumerate(self.action_hyperrects):
            if self.state_in_hyperrect(self.state, hyperrect):
                winning_choice = i
                break
        assert winning_choice != -1

        reward = +1.0 if choice == winning_choice else -1.0
        self.finished = True
        return [reward]

    def is_finished(self):
        return self.finished



