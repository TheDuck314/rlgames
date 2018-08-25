import numpy as np

class Experience:
    """ An Experience is a (state, action, value_est, reward, next_state) tuple from one turn of
    a game for one agent. The agent observed <state>, estimated its value as <value_est>,
    took <action>, and immediately received <reward>. <next_state> is the next step's state
    observation, or None if the episode ended instead. """
    def __init__(self, state, value_est, action, log_p_action, reward, next_state):
        self.state = state
        self.value_est = value_est
        self.action = action
        self.log_p_action = log_p_action
        self.reward = reward
        self.next_state = next_state

    def __repr__(self):
        return "Experience(state={}, value_est={} action={}, log_p_action={}, reward={}, next_state={})".format(self.state, self.value_est, self.action, self.log_p_action, self.reward, self.next_state)


class Episode:
    """ An epside is a list of one agent's experiences from an instance of a game. """
    def __init__(self):
        self.experiences = []

    def __repr__(self):
        return "Episode(experiences={})".format(self.experiences)

    def compute_cum_future_rewards(self):
        """ Given a sequence of experiences from a single episode, compute
        an np.array of floats giving the total cumulative future reward starting
        from each experience. 
        
        E.g. experience returns [1, 2, 3] -> we return np.array([3, 5, 6])
        """
        arr_rewards = np.array([exp.reward for exp in self.experiences])
        return arr_rewards[::-1].cumsum()[::-1]

    def compute_cum_discounted_future_rewards(self, gamma):
        """ Given a sequence of experiences from a single episode, compute
        an np.array of floats giving the total cumulative *discounted* future reward starting
        from each experience. 
        
        E.g. experience returns [1, 2, 3] -> we return np.array([3, 2 + 3*gamma, 1 + 2*gamma + 3*gamma*gamma])
        """
        arr_rewards = np.zeros(len(self.experiences))
        cum_rew = 0.0
        for i in range(len(self.experiences)-1, -1, -1):
            cum_rew *= gamma
            cum_rew += self.experiences[i].reward
            arr_rewards[i] = cum_rew
        return arr_rewards

    def compute_bellman_residuals(self, gamma):
        ret = np.zeros(len(self.experiences))
        for i in range(len(self.experiences)):
            ret[i] = -self.experiences[i].value_est + self.experiences[i].reward
            if i < len(self.experiences) - 1:
                ret[i] += gamma * self.experiences[i+1].value_est
            # else if i is instead the end of the array, there is no next state, which
            # is the same as the next state having value zero, so we don't have to do
            # anything
        return ret

    def compute_generalized_advantage_ests(self, gamma, _lambda):
        """ See https://arxiv.org/pdf/1506.02438.pdf
        We compute the Bellman residuals
          delta_i = -<value_est of state_i> + (reward of action i) + gamma * (value est of state i+1)
        And then the generalized advantage estimator is
          A_i = sum_{l=0}^\infty [(gamma * lambda)^l delta_{i+l}]
        After computing the Bellman residuals this can be computed in one backward pass through the array

        gamma is the standard time discount factor

        intuitively, 1/(1-lambda) is the timescale for a soft cutover from using real future rewards
        to using the value estimate of a future state
        """
        assert 0 <= gamma <= 1
        assert 0 <= _lambda <= 1

        delta = self.compute_bellman_residuals(gamma)
        gamma_lambda = gamma * _lambda

        adv = 0.0
        ret = np.zeros(len(self.experiences))
        for i in range(len(self.experiences)-1, -1, -1):
            adv *= gamma_lambda
            adv += delta[i]
            ret[i] = adv

        return ret

