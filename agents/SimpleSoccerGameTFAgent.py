import tensorflow as tf
import numpy as np
import math

from games.SimpleSoccerGame import SimpleSoccerGame, action_to_int, int_to_action

def linear_layer(inputs, Nin, Nout, name):
    stddev = 0.1 * math.sqrt(2.0 / Nin)
    weights = tf.Variable(tf.truncated_normal([Nin, Nout], stddev=stddev), name=name+"_weights")
    bias = tf.Variable(tf.constant(0.0, shape=[Nout]), name=name+"_biases")
    out = tf.matmul(inputs, weights) + bias
    return out

def relu_fully_connected_layer(inputs, Nin, Nout, name):
    return tf.nn.relu(linear_layer(inputs, Nin, Nout, name), name=name+"_activations")

class SimpleSoccerGameTFAgent:
    def __init__(self, sess):
        self.sess = sess  # tensorflow Session

        self.be_greedy = False

        self.state_size = 14  # (x,y,vx,vy) for agent0, agent1, puck (12) + haspuck0, haspuck1 (2)

        # we output a rank-2 tensor of probabilities of shape (dim, 3)
        self.num_accels = 3  # accel of -1 or 0 or 1 in each direction
        #self.output_size = 2 * self.num_accels + 2  # logit of each accel in each dim, plus logit of (noshoot, shoot)
        self.output_size = self.num_accels**2 * 2  # logit of each accel in each dim, plus logit of (noshoot, shoot)
        assert self.output_size == 18

        self.state_ph = tf.placeholder(tf.float32, shape=[None, self.state_size], name="state")  # shape: (batchsize, 3, dim)

        Nhidden1 = 512
        Nhidden2 = 512
        hidden1 = relu_fully_connected_layer(self.state_ph, Nin=self.state_size, Nout=Nhidden1,  name="hidden1")
        hidden2 = relu_fully_connected_layer(hidden1, Nin=Nhidden1, Nout=Nhidden2,  name="hidden2")

        self.action_logits = linear_layer(hidden2, Nin=Nhidden2, Nout=self.output_size, name="policy_output")
        self.action_dist = tf.distributions.Categorical(logits=self.action_logits)
        self.action_op = self.action_dist.sample()
        self.greedy_action_op = self.action_dist.mode()
        self.log_p_action_op = self.action_dist.log_prob(self.action_op)
        self.chosen_action_ph = tf.placeholder(tf.float32, shape=[None], name="chosen_action")
        self.log_p_chosen_action_op = self.action_dist.log_prob(self.chosen_action_ph)

        self.value_op = tf.reshape(linear_layer(hidden2, Nin=Nhidden2, Nout=1, name="value_output"), [-1])

    def get_log_p_chosen_action_op(self):
        return self.log_p_chosen_action_op
    
    def get_value_op(self):
        return self.value_op

    def states_to_tensor(self, states):
        ret = np.zeros((len(states), self.state_size))
        assert self.state_size == 14
        for i, state in enumerate(states):
            ret[i, 0:2]   = state.agent0.pos
            ret[i, 2:4]   = state.agent0.vel
            ret[i, 4:6]   = state.agent1.pos
            ret[i, 6:8]   = state.agent1.vel
            ret[i, 8:10]  = state.puck.pos
            ret[i, 10:12] = state.puck.vel
            ret[i, 12]    = state.haspuck0
            ret[i, 13]    = state.haspuck1
        return ret

    def make_train_feed_dict(self, experiences):
        return {
            self.state_ph: self.states_to_tensor([exp.state for exp in experiences]),
            self.chosen_action_ph: np.array([action_to_int(exp.action) for exp in experiences])
        }

    def set_be_greedy(self, be_greedy):
        self.be_greedy = be_greedy

    def choose_actions(self, states):
        """ Convert the state to a tensor, feed it in, and sample our action from the output distribution. """
        feed_dict = {self.state_ph: self.states_to_tensor(states)}
        [
            action,
            log_p_action,
            value_est,
        ] = self.sess.run([
            self.greedy_action_op if self.be_greedy else self.action_op,
            self.log_p_action_op,
            self.value_op,
        ], feed_dict=feed_dict)
        assert action.shape == (len(states),)
        assert log_p_action.shape == (len(states),)
        assert value_est.shape == (len(states),)
        ret = []
        for i in range(len(states)):
            one_action = int_to_action(action[i])
            one_log_p_action = log_p_action[i]
            one_value_est = value_est[i]
            ret.append((one_action, one_log_p_action, one_value_est))
        return ret

    def print_debug_info(self):
        pass


