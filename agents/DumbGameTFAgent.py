import tensorflow as tf
import numpy as np

from games import DumbGame

class DumbGameTFAgent:
    def __init__(self, sess):
        self.sess = sess  # tensorflow Session
        self.be_greedy = False

        # model taking state to action probabilities. 
        # (in this case there's no real state)
        self.choice_logits = tf.Variable(tf.constant(0.0, shape=[1, DumbGame.DumbGame.get_num_actions()]))
        self.action_dist = tf.distributions.Categorical(logits=self.choice_logits)
        self.action_op = self.action_dist.sample()
        self.greedy_action_op = self.action_dist.mode()
        self.log_p_action_op = self.action_dist.log_prob(self.action_op)
        self.log_p_greedy_action_op = self.action_dist.log_prob(self.greedy_action_op)

        self.value_op = tf.Variable(tf.constant(0.0, shape=[1]))

        # these will be specified during the update step
        self.chosen_action_ph = tf.placeholder(tf.float32, shape=[None], name='chosen_action')  # shape: (batch size,)
        self.log_p_chosen_action = self.action_dist.log_prob(self.chosen_action_ph)

    def set_be_greedy(self, be_greedy):
        self.be_greedy = be_greedy

    def get_log_p_chosen_action_op(self):
        return self.log_p_chosen_action

    def get_value_op(self):
        return self.value_op

    def _choose_action(self, state):
        # in a non-dumb game we'd have to feed the state into the model
        [
            choice,
            log_p_action,
            value_est
        ] = self.sess.run([
            self.greedy_action_op if self.be_greedy else self.action_op,
            self.log_p_greedy_action_op if self.be_greedy else self.log_p_action_op,
            self.value_op
        ])
        action = DumbGame.Action(int(choice[0]))
        log_p_action = log_p_action[0]
        value_est = value_est[0]
        return action, log_p_action, value_est

    def choose_actions(self, states):
        return map(self._choose_action, states)

    def make_train_feed_dict(self, experiences):
        return {
            self.chosen_action_ph: np.array([exp.action.choice for exp in experiences])
        }

    def print_debug_info(self):
        print "choice_logits =\n{}".format(self.sess.run(self.choice_logits))
        print "value_op =\n{}".format(self.sess.run(self.value_op))



