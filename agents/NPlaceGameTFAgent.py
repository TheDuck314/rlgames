import tensorflow as tf
import numpy as np
import math

from NPlaceGame import *

class NPlaceGameTFAgent:
    def __init__(self, sess):
        self.sess = sess  # tensorflow Session
        self.be_greedy = False

        self.state_size = NPlaceGame.get_num_locations()

        # state will be a one-hot vector indicating our location
        self.state_ph = tf.placeholder(tf.float32, shape=[None, self.state_size], name="state")  # shape: (batchsize, state_size)

        # policy
        self.policy_coefs = tf.Variable(tf.truncated_normal([self.state_size, NPlaceGame.get_num_actions()], stddev=0.01))
        self.choice_logits = tf.matmul(self.state_ph, self.policy_coefs)
        self.action_dist = tf.distributions.Categorical(logits=self.choice_logits)
        self.action_op = self.action_dist.sample()
        self.greedy_action_op = self.action_dist.mode()
        self.log_p_action_op = self.action_dist.log_prob(self.action_op)
        self.log_p_greedy_action_op = self.action_dist.log_prob(self.greedy_action_op)

        # value function
        self.value_coefs = tf.Variable(tf.truncated_normal([self.state_size, 1], stddev=0.01))  # shape: (state_size, 1)
        self.value_op = tf.reshape(tf.matmul(self.state_ph, self.value_coefs), [-1])

        # these will be specified during the update step
        self.chosen_action_ph = tf.placeholder(tf.float32, shape=[None], name='chosen_action')
        self.log_p_chosen_action = self.action_dist.log_prob(self.chosen_action_ph)

    def set_be_greedy(self, be_greedy):
        self.be_greedy = be_greedy

    def get_log_p_chosen_action_op(self):
        return self.log_p_chosen_action

    def get_value_op(self):
        return self.value_op

    def states_to_tensor(self, states):
        """ convert a list of states to a rank-2 tensor where each row is a one-hot vector indicating the location. """
        ret = np.zeros((len(states), self.state_size))
        for i, state in enumerate(states):
            ret[i, state.location] = 1.0
        return ret

    def make_train_feed_dict(self, experiences):
        return {
            self.state_ph: self.states_to_tensor([exp.state for exp in experiences]),
            self.chosen_action_ph: np.array([exp.action.move for exp in experiences]),
        }

    def choose_action(self, state):
        feed_dict = {
            self.state_ph: self.states_to_tensor([state])
        }
        [
            choice,
            log_p_action,
            value_est
        ] = self.sess.run([
            self.greedy_action_op if self.be_greedy else self.action_op,
            self.log_p_greedy_action_op if self.be_greedy else self.log_p_action_op,
            self.value_op
        ], feed_dict=feed_dict)
        assert choice.shape == (1,), "choice = {}".format(choice)
        assert value_est.shape == (1,), "value_est = {}".format(value_est)

        action = NPlaceGameAction(choice[0])
        log_p_action = log_p_action[0]  # one element array -> float
        value_est = value_est[0]  # one element array -> float
        return action, log_p_action, value_est

    def print_debug_info(self):
        np.set_printoptions(precision=3, suppress=True)
        print "policy_coefs =\n{}".format(self.sess.run(self.policy_coefs))
        print "value_coefs =\n{}".format(self.sess.run(self.value_coefs))


