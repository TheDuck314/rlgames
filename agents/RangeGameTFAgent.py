import tensorflow as tf
import numpy as np
import math

from RangeGame import *

def linear_layer(inputs, Nin, Nout):
    stddev = math.sqrt(2.0 / Nin)
    weights = tf.Variable(tf.truncated_normal([Nin, Nout], stddev=stddev))
    bias = tf.Variable(tf.constant(0.0, shape=[Nout]))
    out = tf.matmul(inputs, weights) + bias
    return out

def tanh_fully_connected_layer(inputs, Nin, Nout):
    return tf.tanh(linear_layer(inputs, Nin, Nout))

class RangeGameTFAgent:
    def __init__(self, sess):
        self.sess = sess  # tensorflow Session

        self.state_size = RangeGameState.dim()
        self.num_actions = RangeGame.get_num_actions()

        # model taking state to action probabilities. 
        self.state_numbers_ph = tf.placeholder(tf.float32, shape=[None, self.state_size], name="state_numbers")  # shape: (batchsize, state_size)

        """
        # let's do a linear function from state to logits
        self.linear_coefs = tf.Variable(tf.truncated_normal([self.state_size, self.num_actions], stddev=0.01))  # shape: (state_size, num_actions)
        # (batchsize, state_size) x (state_size, num_actions) -> (batchsize, num_actions)
        self.choice_logits = tf.matmul(self.state_numbers_ph, self.linear_coefs)
        """

        # let's do a one-hidden-layer perceptron
        Nhidden = 32
        #Nhidden = 128
        #Nhidden = 1
        self.hidden = tanh_fully_connected_layer(self.state_numbers_ph, Nin=self.state_size, Nout=Nhidden)  # shape: (batchsize, Nhidden)
        self.choice_logits = linear_layer(self.hidden, Nin=Nhidden, Nout=self.num_actions)  # shape: (batch_size, self.num_actions)
        self.choice_probs = tf.nn.softmax(self.choice_logits)  # shape: (batch_size, self.num_actions)

        # these will be specified during the update step
        self.chosen_action_ph = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='chosen_action')  # shape: (batch size, self.num_actions)
        self.reward_ph = tf.placeholder(tf.float32, shape=[None], name='reward')  # shape: (batch size,)
        # one-hot vector specifying the actually-chosen action
        self.log_choice_probs = tf.log(self.choice_probs)  # shape: (batch_size, self.num_actions)
        # shapes (batch_size, self.num_actions) pointwise mult (batch_size, self.num_actions)  ->  (batch_size, self.num_actions) -> (batch_size,)
        # there's probably a better way to do this selection
        self.log_chosen_action_prob = tf.reduce_sum(self.chosen_action_ph * self.log_choice_probs, reduction_indices=[1])
        # The policy gradient update does one step of gradient descent on
        # this "loss function".
        # (batch_size,) pointwise mult (batch_size,) -> (batch_size,)
        self.reward_times_log_chosen_action_prob = self.reward_ph * self.log_chosen_action_prob
        # (batch_size,) -> scalar
        self.loss_to_minimize = tf.constant(-1.0) * tf.reduce_mean(self.reward_times_log_chosen_action_prob)
        self.learning_rate_ph = tf.placeholder(tf.float32)
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate_ph).minimize(self.loss_to_minimize)
        self.mean_reward = tf.reduce_mean(self.reward_ph)


    def choose_action(self, state):
        # in a real game we'd have to feed the state into the model
        assert state.numbers.shape == (self.state_size, ), "state.numbers.shape = {}".format(state.numbers.shape)
        feed_dict = {
            self.state_numbers_ph: state.numbers.reshape(1, self.state_size)
        }
        choice_prob_values = self.sess.run(self.choice_probs, feed_dict=feed_dict)
        assert choice_prob_values.shape == (1, self.num_actions), "choice_prob_values = {}".format(choice_prob_values)
        choice_prob_values = choice_prob_values.reshape((self.num_actions,))
        choice = np.random.choice(self.num_actions, p=choice_prob_values)
        return RangeGameAction(choice)


    def get_params_str(self):
        [choice_logit_values, choice_prob_values] = self.sess.run([self.choice_logits, self.choice_probs])
        return "DumbTFAgent(choice_logits=({}), choice_probs=({}))".format(choice_logit_values, choice_prob_values)


    def do_policy_gradient_update(self, experience_batch):
        batch_size = len(experience_batch)
        print "batch_size = {}".format(batch_size)
        print "type of batch_size = {}".format(type(batch_size))

        # (batch_size, state dim) states
        state = np.zeros((batch_size, self.state_size))

        # (batch_size, self.num_actions) one-hot vector of chosen actions
        chosen_action = np.zeros((batch_size, self.num_actions))

        # (batch_size,) vector of rewards
        reward = np.zeros(batch_size)

        for i, exp in enumerate(experience_batch):
            state[i, :] = exp.state.numbers

            choice = exp.action.choice
            chosen_action[i, choice] = 1.0

            reward[i] = exp.reward

            # in a real game we'd also need to make a tensor of states

        print "state tensor:\n{}".format(state)
        print "chosen_action tensor:\n{}".format(chosen_action)
        print "reward tensor:\n{}".format(reward)

        #prev_linear_coefs = self.sess.run(self.linear_coefs)
        #print "prev_linear_coefs = {}".format(prev_linear_coefs)

        feed_dict = {
            self.state_numbers_ph: state,
            self.chosen_action_ph: chosen_action,
            self.reward_ph: reward,
            self.learning_rate_ph: 0.5
        }

        [
            _, 
            choice_probs_value,
            log_choice_probs_value,
            log_chosen_action_prob_value,
            reward_times_log_chosen_action_prob_value,
            loss_value,
            mean_reward_value
        ] = self.sess.run([
            self.train_op, 
            self.choice_probs, 
            self.log_choice_probs, 
            self.log_chosen_action_prob, 
            self.reward_times_log_chosen_action_prob,
            self.loss_to_minimize,
            self.mean_reward
        ], feed_dict=feed_dict)

        print "choice_probs_value = {}".format(choice_probs_value)
        print "log_choice_probs_value = {}".format(log_choice_probs_value)
        print "log_chosen_action_prob_value = {}".format(log_chosen_action_prob_value)
        print "reward_times_log_chosen_action_prob_value = {}".format(reward_times_log_chosen_action_prob_value)
        print "loss_value = {}".format(loss_value)
        print "mean_reward_value = {}".format(mean_reward_value)

        for i in range(self.num_actions):
            print "rewards from choice {}: {}".format(i, [exp.reward for exp in experience_batch if exp.action.choice == i])

        #new_linear_coefs = self.sess.run(self.linear_coefs)
        #print "new_linear_coefs = {}".format(new_linear_coefs)


