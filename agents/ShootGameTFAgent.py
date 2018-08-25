import tensorflow as tf
import numpy as np
import math

from ShootGame import *

def linear_layer(inputs, Nin, Nout, name):
    stddev = 0.1 * math.sqrt(2.0 / Nin)
    #stddev = 0.03
    weights = tf.Variable(tf.truncated_normal([Nin, Nout], stddev=stddev), name=name+"_weights")
    bias = tf.Variable(tf.constant(0.0, shape=[Nout]), name=name+"_biases")
    out = tf.matmul(inputs, weights) + bias
    return out

def relu_fully_connected_layer(inputs, Nin, Nout, name):
    return tf.nn.relu(linear_layer(inputs, Nin, Nout, name), name=name+"_activations")

class ShootGameTFAgent:
    def __init__(self, sess):
        self.sess = sess  # tensorflow Session
        np.set_printoptions(precision=3, suppress=True)

        self.be_greedy = False

        self.dim = ShootGame.dim
        self.state_size = self.dim * 4 + 1  # = 9: our (x, y, vx, vy), puck (x, y, vx, vy), haspuck

        # we output a rank-2 tensor of probabilities of shape (dim, 3)
        self.num_accels_per_dim = 3  # accel of -1 or 0 or 1 in each direction
        self.output_size = self.dim * self.num_accels_per_dim + 2  # logit of each accel in each dim, plus logit of (noshoot, shoot)

        self.state_ph = tf.placeholder(tf.float32, shape=[None, self.state_size], name="state")  # shape: (batchsize, 3, dim)

        # 1 hidden layer perceptron
        policy_Nhidden = 2048
        policy_hidden = relu_fully_connected_layer(self.state_ph, Nin=self.state_size, Nout=policy_Nhidden, name="policy_hidden")
        self.action_logits = linear_layer(policy_hidden, Nin=policy_Nhidden, Nout=self.output_size, name="policy_output")

        # slice up the flat array of logits into the separate decisions about acceleration and shooting
        self.accel_logits = [self.action_logits[:, d*self.num_accels_per_dim:(d+1)*self.num_accels_per_dim] for d in range(self.dim)]
        self.shoot_logits = self.action_logits[:, self.dim*self.num_accels_per_dim:]
        # convert logits for each decision into probabilities
        self.accel_probs = [tf.nn.softmax(logits) for logits in self.accel_logits]
        # for the shooting probs, we enforce a 100% chance of not shooting if we
        # don't actually have the puck. I think this should help training. 
        self.can_shoot = tf.reshape(self.state_ph[:, self.dim*4], [-1, 1])  # haspuck. reshape to (batch_size, 1)
        self.cant_shoot_fixed_probs = tf.constant([[1.0, 0.0]])  # 0% chance of shooting
        self.shoot_probs = tf.nn.softmax(self.shoot_logits) * self.can_shoot + self.cant_shoot_fixed_probs * (1.0 - self.can_shoot)

        # these will be specified during the update step
        # one-hot:
        self.chosen_accel_phs = [tf.placeholder(tf.float32, shape=[None, self.num_accels_per_dim], name='chosen_accel_{}'.format(d)) for d in range(self.dim)]
        # one-hit: 
        self.chosen_shoot_ph = tf.placeholder(tf.float32, shape=[None, 2], name='chosen_shoot'.format(d))

        assert self.dim == 2 
        self.log_chosen_action_prob = (tf.log(tf.reduce_sum(self.chosen_accel_phs[0] * self.accel_probs[0], reduction_indices=[1])) +
                                       tf.log(tf.reduce_sum(self.chosen_accel_phs[1] * self.accel_probs[1], reduction_indices=[1])) + 
                                       tf.log(tf.reduce_sum(self.chosen_shoot_ph * self.shoot_probs, reduction_indices=[1])))

        # value function
        # 2 hidden layer perceptron
        #value_Nhidden_1 = 16
        #value_hidden_1 = tanh_fully_connected_layer(self.state_ph, Nin=self.state_size, Nout=value_Nhidden_1, name="value_hidden_1")
        #value_Nhidden_2 = 16
        #value_hidden_2 = tanh_fully_connected_layer(value_hidden_1, Nin=value_Nhidden_1, Nout=value_Nhidden_2, name="value_hidden_2")
        #self.value_op = tf.tanh(tf.reshape(linear_layer(value_hidden_2, Nin=value_Nhidden_2, Nout=1, name="value_final"), [-1]), name="value_op")

        #puck_velocity = self.state_ph[:, 3*self.dim:4*self.dim]
        #puck_speed = tf.sqrt(tf.reduce_sum(tf.square(puck_velocity), reduction_indices=[1]))
        #self.puck_speed_coef = tf.Variable(tf.constant(0.0, shape=[1]), name="puck_speed_coef")
        #self.puck_speed_intercept = tf.Variable(tf.constant(0.0, shape=[1]), name="puck_speed_intercept")
        #self.value_op = puck_speed * self.puck_speed_coef + self.puck_speed_intercept

        #self.value_linear_coefs = tf.Variable(tf.truncated_normal([self.state_size, 1], stddev=0.01), name="value_linear_coefs")
        #self.value_bias = tf.Variable(tf.constant(0.0), name="value_bias")
        #self.value_op = tf.reshape(tf.matmul(tf.abs(self.state_ph), self.value_linear_coefs) + self.value_bias, [-1], name="value_op")

        value_Nhidden_1 = 64
        value_hidden_1 = relu_fully_connected_layer(self.state_ph, Nin=self.state_size, Nout=value_Nhidden_1, name="value_hidden_1")
        self.value_op = tf.reshape(linear_layer(value_hidden_1, Nin=value_Nhidden_1, Nout=1, name="value_output"), [-1])


    def get_log_p_chosen_action_op(self):
        return self.log_chosen_action_prob

    
    def get_value_op(self):
        return self.value_op


    def make_train_feed_dict(self, experiences):
        batch_size = len(experiences)

        states = self.states_to_tensor([exp.state for exp in experiences])

        # (batch_size, self.num_actions) one-hot vector of chosen actions
        chosen_accels = np.zeros((batch_size, self.dim, self.num_accels_per_dim))
        chosen_shoots = np.zeros((batch_size, 2))
        assert self.num_accels_per_dim == 3
        for i, exp in enumerate(experiences):
            accels = exp.action.accels
            for d in range(self.dim):
                # accel in [-1, 0, 1] gets mapped to [0, 1, 2]
                chosen_accels[i, d, accels[d]+1] = 1.0
            shoot_idx = 0 if not exp.action.shoot else 1
            chosen_shoots[i, shoot_idx] = 1.0

        return {
            self.state_ph: states,
            self.chosen_accel_phs[0]: chosen_accels[:, 0, :],
            self.chosen_accel_phs[1]: chosen_accels[:, 1, :],
            self.chosen_shoot_ph: chosen_shoots,
        }


    def states_to_tensor(self, states):
        """ convert a list of states to vectors
        """
        ret = np.zeros((len(states), self.state_size))
        for i, state in enumerate(states):
            ret[i, 0*self.dim:1*self.dim] = state.agent.position
            ret[i, 1*self.dim:2*self.dim] = state.agent.velocity
            ret[i, 2*self.dim:3*self.dim] = state.puck.position
            ret[i, 3*self.dim:4*self.dim] = state.puck.velocity
            ret[i, 4*self.dim] = 1.0 if state.haspuck else 0.0
        return ret


    def set_be_greedy(self, be_greedy):
        """ If be_greedy is true, we'll always pick the action with the highest
        probability. If it's false, we'll sample from the probability
        distribution. """
        self.be_greedy = be_greedy


    def choose_action(self, state):
        """ Convert the state to a tensor, feed it in, and sample our action from the output distribution. """
        feed_dict = {
            self.state_ph: self.states_to_tensor([state])
        }
        assert self.dim == 2
        [
            accel_probs_x_values,
            accel_probs_y_values,
            shoot_probs_values,
            value_est,
        ] = self.sess.run([
            self.accel_probs[0],
            self.accel_probs[1],
            self.shoot_probs,
            self.value_op,
        ], feed_dict=feed_dict)
        assert accel_probs_x_values.shape == (1, self.num_accels_per_dim)
        assert accel_probs_y_values.shape == (1, self.num_accels_per_dim)
        assert shoot_probs_values.shape == (1, 2)
        assert value_est.shape == (1,)
        accel_probs_values = [accel_probs_x_values, accel_probs_y_values]

        accels = np.zeros(self.dim, dtype=np.int32)
        for d in range(self.dim):
            if self.be_greedy:
                # take highest-probability choice
                choice = np.argmax(accel_probs_values[d][0, :])
            else:
                # sample choice from the distribution
                choice = np.random.choice(self.num_accels_per_dim, p=accel_probs_values[d][0, :])
            assert self.num_accels_per_dim == 3
            accels[d] = choice - 1

        if self.be_greedy:
            shoot = shoot_probs_values[0, 1] > 0.5
        else:
            shoot = np.random.choice([False, True], p=shoot_probs_values[0, :])

        action = ShootGameAction(accels=accels, shoot=shoot)

        return action, value_est[0]
#        return action, 0.0


    def prepare_episode(self, episode):
        """ Given an episode of experiences, make some tensors that will be fed
        in for training. The first dimension of the tensors indexes over
        experiences.
        """
        ep_len = len(episode.experiences)

        # (ep_len, state_size) states
        state = self.states_to_tensor([exp.state for exp in episode.experiences])

        # (ep_len, self.num_actions) one-hot vector of chosen actions
        chosen_accels = np.zeros((ep_len, self.dim, self.num_accels_per_dim))
        chosen_shoots = np.zeros((ep_len, 2))
        assert self.num_accels_per_dim == 3
        for i, exp in enumerate(episode.experiences):
            accels = exp.action.accels
            for d in range(self.dim):
                # accel in [-1, 0, 1] gets mapped to [0, 1, 2]
                chosen_accels[i, d, accels[d]+1] = 1.0
            shoot_idx = 0 if not exp.action.shoot else 1
            chosen_shoots[i, shoot_idx] = 1.0

        # (ep_len,) vector of rewards
        reward = episode.compute_cum_discounted_future_rewards(gamma=0.95)

        return state, chosen_accels, chosen_shoots, reward


    def do_policy_gradient_update(self, episodes):
        batch_size = sum(len(ep.experiences) for ep in episodes)

        # make some big tensors out of all the episodes
        prepared_tensors = [self.prepare_episode(ep) for ep in episodes]
        states, chosen_accels, chosen_shoots, rewards = [np.concatenate(tensor_list) for tensor_list in zip(*prepared_tensors)]
        #print "states =\n{}".format(states)
        #print "chosen_accels =\n{}".format(chosen_accels)
        #print "chosen_shoots =\n{}".format(chosen_shoots)
        #print "rewards =\n{}".format(rewards)
        assert states.shape == (batch_size, self.state_size)
        assert chosen_accels.shape == (batch_size, self.dim, self.num_accels_per_dim)
        assert chosen_shoots.shape == (batch_size, 2)
        assert rewards.shape == (batch_size,)

        #prev_linear_coefs = self.sess.run(self.linear_coefs)
        #print "prev_linear_coefs = {}".format(prev_linear_coefs)

        assert self.dim == 2
        feed_dict = {
            self.state_ph: states,
            self.chosen_accel_phs[0]: chosen_accels[:, 0, :],
            self.chosen_accel_phs[1]: chosen_accels[:, 1, :],
            self.chosen_shoot_ph: chosen_shoots,
            self.reward_ph: rewards,
            self.policy_learning_rate_ph: 0.001,
        }

        [
            _, 
            action_logits_value,
            accel_logits_x_value,
            accel_logits_y_value,
            shoot_logits_value,
            accel_probs_x_value,
            accel_probs_y_value,
            can_shoot_value,
            cant_shoot_fixed_probs_value,
            shoot_probs_value,
            log_chosen_action_prob_value,
#            advantage_thing_value,
            policy_loss_to_minimize_value,
            mean_reward_value,
        ] = self.sess.run([
            self.policy_train_op, 
            self.action_logits,
            self.accel_logits[0],
            self.accel_logits[1],
            self.shoot_logits,
            self.accel_probs[0],
            self.accel_probs[1],
            self.can_shoot,
            self.cant_shoot_fixed_probs,
            self.shoot_probs,
            self.log_chosen_action_prob, 
#            self.advantage_thing,
            self.policy_loss_to_minimize,
            self.mean_reward,
        ], feed_dict=feed_dict)

        print "action_logits_value =\n{}".format(action_logits_value)
        print "accel_logits_x_value =\n{}".format(accel_logits_x_value)
        print "accel_logits_y_value =\n{}".format(accel_logits_y_value)
        print "shoot_logits_value =\n{}".format(shoot_logits_value)
        print "accel_probs_x_value =\n{}".format(accel_probs_x_value)
        print "accel_probs_y_value =\n{}".format(accel_probs_y_value)
        print "can_shoot_value =\n{}".format(can_shoot_value)
        print "cant_shoot_fixed_probs_value =\n{}".format(cant_shoot_fixed_probs_value)
        print "shoot_probs_value =\n{}".format(shoot_probs_value)
        print "log_chosen_action_prob_value =\n{}".format(log_chosen_action_prob_value)
#        print "advantage_thing_value =\n{}".format(advantage_thing_value)
        print "policy_loss_to_minimize_value =\n{}".format(policy_loss_to_minimize_value)
        print "mean_reward_value =\n{}".format(mean_reward_value)

        new_linear_coefs = self.sess.run(self.linear_coefs)
        print "new_linear_coefs = {}".format(new_linear_coefs)

    def print_debug_info(self):
        pass
        #print "linear_coefs =\n{}".format(self.sess.run(self.linear_coefs))
        #print "value_linear_coefs =\n{}".format(self.sess.run(self.value_linear_coefs))
        #print "value_bias = {}".format(self.sess.run(self.value_bias))
        #print "puck_speed_coef = {}".format(self.sess.run(self.puck_speed_coef))
        #print "puck_speed_intercept = {}".format(self.sess.run(self.puck_speed_intercept))


