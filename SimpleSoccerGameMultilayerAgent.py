import tensorflow as tf
import numpy as np
import math

from SimpleSoccerGame import *

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
        self.output_size = 2 * self.num_accels + 2  # logit of each accel in each dim, plus logit of (noshoot, shoot)

        self.state_ph = tf.placeholder(tf.float32, shape=[None, self.state_size], name="state")  # shape: (batchsize, 3, dim)

        # 1 hidden layer perceptron
        policy_Nhidden = 2048
        policy_hidden = relu_fully_connected_layer(self.state_ph, Nin=self.state_size, Nout=policy_Nhidden, name="policy_hidden")
        self.action_logits = linear_layer(policy_hidden, Nin=policy_Nhidden, Nout=self.output_size, name="policy_output")

        # slice up the flat array of logits into the separate decisions about acceleration and shooting
        # TODO: this seems limiting. We are forcing our action distribution to factor into parts for
        # accel_x, accel_y, and shoot. It would be more general to just output a distribution over the 3*3*2 = 18
        # possible compound actions.
        self.accel_x_dist = tf.distributions.Categorical(logits=self.action_logits[:, 0:self.num_accels])
        self.accel_y_dist = tf.distributions.Categorical(logits=self.action_logits[:, self.num_accels:2*self.num_accels])
        self.shoot_dist   = tf.distributions.Categorical(logits=self.action_logits[:, 2*self.num_accels:self.state_size])
        self.accel_x_op = self.accel_x_dist.sample()
        self.accel_y_op = self.accel_y_dist.sample()
        self.shoot_op   = self.shoot_dist.sample()
        self.greedy_accel_x_op = self.accel_x_dist.mode()
        self.greedy_accel_y_op = self.accel_y_dist.mode()
        self.greedy_shoot_op   = self.shoot_dist.mode()
        self.log_p_action_op = (self.accel_x_dist.log_prob(self.accel_x_op) + 
                                self.accel_y_dist.log_prob(self.accel_y_op) +
                                self.shoot_dist.log_prob(self.shoot_op))

        # compute log prob of a given action
        self.chosen_accel_x_ph = tf.placeholder(tf.float32, shape=[None], name="chosen_accel_x")
        self.chosen_accel_y_ph = tf.placeholder(tf.float32, shape=[None], name="chosen_accel_y")
        self.chosen_shoot_ph   = tf.placeholder(tf.float32, shape=[None], name='chosen_shoot')
        self.log_p_chosen_action_op = (self.accel_x_dist.log_prob(self.chosen_accel_x_ph) + 
                                       self.accel_y_dist.log_prob(self.chosen_accel_y_ph) + 
                                       self.shoot_dist.log_prob(self.chosen_shoot_ph))

        # value function
        # 1 hidden layer perceptron
        value_Nhidden_1 = 2048
        value_hidden_1 = relu_fully_connected_layer(self.state_ph, Nin=self.state_size, Nout=value_Nhidden_1, name="value_hidden_1")
        self.value_op = tf.reshape(linear_layer(value_hidden_1, Nin=value_Nhidden_1, Nout=1, name="value_output"), [-1])

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
            self.chosen_accel_x_ph: np.array([exp.action.accels[0] for exp in experiences]),
            self.chosen_accel_y_ph: np.array([exp.action.accels[1] for exp in experiences]),
            self.chosen_shoot_ph:   np.array([exp.action.shoot for exp in experiences]),
        }

    def set_be_greedy(self, be_greedy):
        self.be_greedy = be_greedy

    def choose_action(self, state):
        """ Convert the state to a tensor, feed it in, and sample our action from the output distribution. """
        feed_dict = {self.state_ph: self.states_to_tensor([state])}
        [
            accel_x,
            accel_y,
            shoot,
            log_p_action,  # wrong if be_greedy is True, but whatever
            value_est,
        ] = self.sess.run([
            self.greedy_accel_x_op if self.be_greedy else self.accel_x_op,
            self.greedy_accel_y_op if self.be_greedy else self.accel_y_op,
            self.greedy_shoot_op if self.be_greedy else self.shoot_op,
            self.log_p_action_op,
            self.value_op,
        ], feed_dict=feed_dict)
        assert accel_x.shape == (1,)
        assert accel_y.shape == (1,)
        assert shoot.shape == (1,)
        assert log_p_action.shape == (1,)
        assert value_est.shape == (1,)

        action = SimpleSoccerAction(
            accels = np.array([accel_x[0], accel_y[0]]),
            shoot  = shoot[0]
        )
        log_p_action = log_p_action[0]
        value_est = value_est[0]
        return action, log_p_action, value_est


    def print_debug_info(self):
        pass


