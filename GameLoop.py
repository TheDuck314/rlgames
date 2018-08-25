# GameLoop runs a game

import pickle

from DumbGame import *
from Experience import *

class GameResult:
    """ A GameResult contains all the data that results from playing a game:
      - the Episodes containing the experiences of the agents
      - the sequence of "true" states as seen by an omniscient third party,
        rather than the agents
    """
    def __init__(self, num_agents):
        self.episodes = [Episode() for i in range(num_agents)]
        self.true_states = []

    def __repr__(self):
        return "GameResult(episodes={}, true_states={})".format(self.episodes, self.true_states)

    def save(self, fn):
        """ Just pickle ourselves to the file """
        assert fn.endswith(".pkl")
        with open(fn, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fn):
        assert fn.endswith(".pkl")
        with open(fn, "rb") as f:
            return pickle.load(f)


def play_game(game, agents):
    """ Given a game and some agents, run the game step by step until it's finished. 
    Return a GameResult. """
    num_agents = game.get_num_agents()
    assert num_agents == len(agents), "{} != {}".format(num_agents, len(agents))

    result = GameResult(num_agents=num_agents)

    # capture the initial game state
    result.true_states.append(game.get_true_state())

    while not game.is_finished():
        # Get the state observed by each of the agents
        states = game.get_observed_states()

        # If this isn't the first step, set the <next_state> field of the
        # previous experience for each agent
        for st, ep in zip(states, result.episodes):
            if ep.experiences:
                ep.experiences[-1].next_state = st

        assert len(states) == len(agents)
        # ask each agent for an action and an estimated value of this state
        actions = []
        log_p_actions = []
        value_ests = []
        for agent, state in zip(agents, states):
            action, log_p_action, value_est = agent.choose_action(state)
            actions.append(action)
            log_p_actions.append(log_p_action)
            value_ests.append(value_est)
        # give the model the actions, and have it simulate a step
        # and return any immediate rewards
        rewards = game.simulate_step(actions)

        # record the Experience of each agent on this time step
        for i in range(num_agents):
            result.episodes[i].experiences.append(Experience(
                state        = states[i],
                value_est    = value_ests[i],
                action       = actions[i],
                log_p_action = log_p_actions[i],
                reward       = rewards[i],
                next_state   = None,  # will be set in next iteration, if there is one
            ))

        result.true_states.append(game.get_true_state())

    return result
