import tensorflow as tf
import numpy as np
import math

from NPlaceGame import *

class NPlaceGameQAgent:
    def __init__(self, sess):
        self.sess = sess  # tensorflow Session
        self.epsilon = 0.0

        self.state_size = NPlaceGame.num_locations()
        self.num_actions = 3

        # state will be a one-hot vector indicating our location
        self.state_ph = tf.placeholder(tf.float32, shape=[None, self.state_size], name="state")  # shape: (batchsize, state_size)

        # let's do a linear function from state to Q value of actions
        self.linear_coefs = tf.Variable(tf.truncated_normal([self.state_size, self.num_actions], stddev=0.01))  # shape: (state_size, num_actions)
        # (batchsize, state_size) x (state_size, num_actions) -> (batchsize, num_actions)
        self.q_op = tf.matmul(self.state_ph, self.linear_coefs)

        # these will be specified during the update step
        self.chosen_action_ph = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='chosen_action')  # shape: (batch size, self.num_actions)
        self.q_target_ph = tf.placeholder(tf.float32, shape=[None], name='chosen_action')  # shape: (batch size, self.num_actions)

        # use tf.gather_nd after updating tensorflow
        self.q_of_chosen_action = tf.reduce_sum(self.chosen_action_ph * self.q_op, reduction_indices=[1])
        self.q_mse = tf.reduce_mean(tf.square(self.q_of_chosen_action - self.q_target_ph))
        self.learning_rate_ph = tf.placeholder(tf.float32)
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate_ph).minimize(self.q_mse)


    def states_to_onehot(self, states):
        """ convert a list of states to a rank-2 tensor where each row is a one-hot vector indicating the location. """
        ret = np.zeros((len(states), self.state_size))
        for i, state in enumerate(states):
            # we have to handle state = None. (but it doesn't matter what the Q
            # function outputs in this case). Just leave the vector as zero
            if state is not None:
                ret[i, state.location] = 1.0
        return ret


    def set_epsilon(self, epsilon):
        """ epsilon is the probability of choosing a random action. """
        self.epsilon = epsilon


    def choose_action_and_value_est(self, state):
        """ Convert the state to a one-hot location vector, feed it in, and sample our move from the output distribution. """
        feed_dict = {
            self.state_ph: self.states_to_onehot([state])
        }
        [
            q_values,
        ] = self.sess.run([
            self.q_op, 
        ], feed_dict=feed_dict)
        assert q_values.shape == (1, self.num_actions), "q_values = {}".format(choice_prob_values)
        q_values = q_values.reshape((self.num_actions,))


        if np.random.rand() > self.epsilon:
            # choose the action with highest q-value
            choice = np.argmax(q_values)
        else:
            # choose an action at random
            choice = np.random.randint(self.num_actions)

        move = choice - 1  # map [0, 1, 2] -> [-1, 0, 1]
        action = NPlaceGameAction(move)

        # value_est of the state won't be used by regular Q-learning, but it's
        # easy to compute so let's just return it.
        value_est = np.max(q_values)

        return action, value_est


    def prepare_minibatch(self, experiences):
        """ Given an minibatch of experiences, make some tensors that will be fed
        in for training. The first dimension of the tensors indexes over
        experiences. """
        batch_size = len(experiences)

        # (batch_size, state dim) states
        # for NPlaceGame these are 1-hot vectors representing the location
        states = self.states_to_onehot([exp.state for exp in experiences])

        # (batch_size, state dim) states
        # for NPlaceGame these are 1-hot vectors representing the location
        next_state_exists = np.array([exp.next_state is not None for exp in experiences])
        next_states = self.states_to_onehot([exp.next_state for exp in experiences])

        # (batch_size, self.num_actions) one-hot vector of chosen actions
        chosen_actions = np.zeros((batch_size, self.num_actions))
        for i, exp in enumerate(experiences):
            move = exp.action.move
            chosen_actions[i, move+1] = 1.0

        # (batch_size,) simple one-step rewards
        rewards = np.array([exp.reward for exp in experiences])

        return states, next_state_exists, next_states, chosen_actions, rewards


    def do_q_update(self, experiences, gamma):
        batch_size = len(experiences)
        print "batch_size = {}".format(batch_size)

        # make some big tensors out of all the experiences;
        states, next_state_exists, next_states, chosen_actions, rewards = self.prepare_minibatch(experiences)
        print "states =\n{}".format(states)
        print "next_state_exists =\n{}".format(next_state_exists)
        print "next_states =\n{}".format(next_states)
        print "chosen_actions =\n{}".format(chosen_actions)
        print "rewards =\n{}".format(rewards)
        assert states.shape == (batch_size, self.state_size)
        assert next_state_exists.shape == (batch_size,)
        assert next_states.shape == (batch_size, self.state_size)
        assert chosen_actions.shape == (batch_size, self.num_actions)
        assert rewards.shape == (batch_size,)

        prev_linear_coefs = self.sess.run(self.linear_coefs)
        print "prev_linear_coefs = {}".format(prev_linear_coefs)

        # For Q-learning we do gradient descent on 
        #  (q_targets - Q)^2
        # where q_targets is (reward + gamma * [max over actions of Q values of next_state])
        # So we need to compute the Q values of the next_state.
        # One problem is that there might be no next state, if the game ended after the
        # current state. For that we input a dummy state vector to the Q network, then
        # multiply the resulting Q values by zero
        feed_dict = {self.state_ph: next_states}
        [next_states_q] = self.sess.run([self.q_op], feed_dict=feed_dict)
        assert next_states_q.shape == (batch_size, self.num_actions)
        # next_states_q has shape (batchsize, num_actions)
        # we want the max over actions
        next_states_max_q = np.amax(next_states_q, axis=1)
        assert next_states_max_q.shape == (batch_size,)
        # zero out next state q if the next state doesn't actually exist
        next_states_max_q *= next_state_exists
        q_targets = rewards + gamma * next_states_max_q  # shape: (batch_size,)
        print "next_states_q =\n{}".format(next_states_q)
        print "next_states_max_q =\n{}".format(next_states_max_q)
        print "q_targets =\n{}".format(q_targets)

        feed_dict = {
            self.state_ph: states,
            self.chosen_action_ph: chosen_actions,
            self.q_target_ph: q_targets,
            self.learning_rate_ph: 1.0,
        }

        [
            _,
            q_values,
            q_of_chosen_action_values,
            q_mse_value,
        ] = self.sess.run([
            self.train_op, 
            self.q_op, 
            self.q_of_chosen_action,
            self.q_mse,
        ], feed_dict=feed_dict)

        print "q_values = {}".format(q_values)
        print "q_of_chosen_action_values = {}".format(q_of_chosen_action_values)
        print "q_mse_value = {}".format(q_mse_value)

        np.set_printoptions(precision=7, suppress=True)
        new_linear_coefs = self.sess.run(self.linear_coefs)
        print "new_linear_coefs = {}".format(new_linear_coefs)


