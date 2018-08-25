import tensorflow as tf
import numpy as np
import math

from SlideGame import *

def linear_layer(inputs, Nin, Nout):
    stddev = math.sqrt(2.0 / Nin)
    weights = tf.Variable(tf.truncated_normal([Nin, Nout], stddev=stddev))
    bias = tf.Variable(tf.constant(0.0, shape=[Nout]))
    out = tf.matmul(inputs, weights) + bias
    return out

def tanh_fully_connected_layer(inputs, Nin, Nout):
    return tf.tanh(linear_layer(inputs, Nin, Nout))

class SlideGameTFAgent:
    def __init__(self, sess):
        self.sess = sess  # tensorflow Session

        self.be_greedy = False

        self.dim = SlideGame.dim()

        # we output a rank-2 tensor of probabilities of shape (dim, 3)
        self.num_actions_per_dim = 2  # accel of -1 or 1 in each direction

        # the state for one instance is a rank-2 tensor of shape (3, dim), e.g. [[posx, posy], [velx, vely], [destx, desty]]
        self.state_ph = tf.placeholder(tf.float32, shape=[None, 3, self.dim], name="state")  # shape: (batchsize, 3, dim)

        # let's do a linear function from state to logits
        #self.linear_coefs = tf.Variable(tf.truncated_normal([3, self.dim, self.dim, self.num_actions_per_dim], stddev=0.01))  # shape: (3, dim, dim, num_actions_per_dim)
        self.linear_coefs = tf.Variable(tf.truncated_normal([3 * self.dim, self.dim * self.num_actions_per_dim], stddev=0.01))  # shape: (3, dim, dim, num_actions_per_dim)

        # (batchsize, 3, dim)_abc contract (3, dim, dim, num_actions_per_dim)_bcde -> (batchsize, dim, num_actions_per_dim)
        #self.choice_logits = tf.tensordot(self.state_ph, self.linear_coefs, [[1, 2], [0, 1]])
        # tensordot isn't in this version :(
        self.choice_logits = tf.reshape(tf.matmul(tf.reshape(self.state_ph, [-1, 3 * self.dim]), 
                                                  self.linear_coefs),
                                        [-1, self.dim, self.num_actions_per_dim])
        #self.choice_probs = tf.nn.softmax(self.choice_logits)  # shape: (batch_size, dim, num_actions_per_dim)
        self.choice_probs = tf.reshape(tf.nn.softmax(tf.reshape(self.choice_logits, [-1, self.num_actions_per_dim])), [-1, self.dim, self.num_actions_per_dim])

        # let's also learn a value function
        """
        self.value_coefs = tf.Variable(tf.truncated_normal([self.state_size, 1], stddev=0.01))  # shape: (state_size, 1)
        # (batchsize, state_size) x (stat_size, 1) -> (batchsize, 1) -> (batchsize,)
        self.value_est_op = tf.reshape(tf.matmul(self.state_ph, self.value_coefs), [-1])
        """

        # a linear value function is not going to work here. Let's try a one-hidden-layer perceptron
        Nhidden = 32
        #Nhidden = 128
        #Nhidden = 1
        # (batchsize, 3, dim) -> (batchsize, 3 * dim)
        flattened_state = tf.reshape(self.state_ph, [-1, 3 * self.dim])  # shape: (batchsize, 3 * dim)
        self.hidden = tanh_fully_connected_layer(flattened_state, Nin=(3 * self.dim), Nout=Nhidden)  # shape: (batchsize, Nhidden)
        # (batchsize, Nhidden) -> (batchsize,)
        self.value_est_op = tf.reduce_sum(self.hidden, reduction_indices=[1])  # shape: (batchsize,)

        # these will be specified during the update step
        self.chosen_action_ph = tf.placeholder(tf.float32, shape=[None, self.dim, self.num_actions_per_dim], name='chosen_action')  # shape: (batch size, dim, num_actions_per_dim)
        self.reward_ph = tf.placeholder(tf.float32, shape=[None], name='reward')  # shape: (batch size,)
        # one-hot vector specifying the actually-chosen action
        self.log_choice_probs = tf.log(self.choice_probs)  # shape: (batch_size, dim, num_actions_per_dim)
        # shapes (batch_size, dim, num_actions_per_dim) pointwise mult (batch_size, dim, num_actions_per_dim)  ->  (batch_size, self.num_actions) -> (batch_size,)
        # there's probably a better way to do this selection
        # Note: we choose an action along each axis. Prob of overall compound action is product of probabilities
        # of axis actions. So log of prop of overall compound action is sum of axis action log probs. So we can
        # just sum over axis 1 (sum over dimensions) as well as axis 2 (sum over actions within dimension)
        self.log_chosen_action_prob = tf.reduce_sum(self.chosen_action_ph * self.log_choice_probs, reduction_indices=[1, 2])
        # The policy gradient update does one step of gradient descent on
        # this "loss function".
        # (batch_size,) pointwise mult (batch_size,) -> (batch_size,)
        # OPTION 1: JUST STICK IN THE TRUE REWARD:
        self.advantage_thing = self.reward_ph
        # OPTION 2: STICK IN REWARD MINUS PREDICTED REWARD OF STATE (VIA VALUE FUNCTION; PREDICTION NOT CONDITIONED ON ACTION)
        #self.advantage_thing = (self.reward_ph - self.value_est_ph)  # note: use ph, not value_est_op, so gradient descent doesn't try to optimize the value function here
        # OPTION 3: STICK IN SIMPLE ADVANTAGE ESTIMATOR
        #self.advantage_thing = tf.placeholder(tf.float32, shape=[None], name="simple_advantage_est")  # shape: (batch_size,)
        # (batch_size,) -> scalar
        self.policy_loss_to_minimize = tf.constant(-1.0) * tf.reduce_mean(self.advantage_thing * self.log_chosen_action_prob)
        self.policy_learning_rate_ph = tf.placeholder(tf.float32)
        self.policy_train_op = tf.train.GradientDescentOptimizer(self.policy_learning_rate_ph).minimize(self.policy_loss_to_minimize)
        self.mean_reward = tf.reduce_mean(self.reward_ph)

        # training of the value function
        # loss for value function is just mean squared error
        self.value_mse = tf.reduce_mean(tf.square(self.reward_ph - self.value_est_op))
        self.value_learning_rate_ph = tf.placeholder(tf.float32)
        self.value_train_op = tf.train.GradientDescentOptimizer(self.value_learning_rate_ph).minimize(self.value_mse)


    def states_to_tensor(self, states):
        """ convert a list of states to a rank-3 tensor:
            - axis 0: index within batch
            - axis 1: 0, 1, 2 to choose pos, vel, dest
            - axis 2: components of vector (length dim)
        """
        ret = np.zeros((len(states), 3, self.dim))
        for i, state in enumerate(states):
            ret[i, 0, :] = state.position
            ret[i, 1, :] = state.velocity
            ret[i, 2, :] = state.dest
        return ret


    def set_be_greedy(self, be_greedy):
        """ If be_greedy is true, we'll always pick the action with the highest
        probability. If it's false, we'll sample from the probability
        distribution. """
        self.be_greedy = be_greedy


    def choose_action_and_value_est(self, state):
        """ Convert the state to a tensor, feed it in, and sample our action from the output distribution. """
        feed_dict = {
            self.state_ph: self.states_to_tensor([state])
        }
        [
            choice_prob_values,
            value_est,
        ] = self.sess.run([
            self.choice_probs, 
            self.value_est_op,
        ], feed_dict=feed_dict)
        assert choice_prob_values.shape == (1, self.dim, self.num_actions_per_dim), "choice_prob_values = {}".format(choice_prob_values)
        assert value_est.shape == (1,), "value_est = {}".format(value_est)

        choice_prob_values = choice_prob_values.reshape((self.dim, self.num_actions_per_dim))
        accels = np.zeros(self.dim)
        for d in range(self.dim):
            if self.be_greedy:
                # take highest-probability choice
                choice = np.argmax(choice_prob_values[d, :])
            else:
                # sample choice from the distribution
                choice = np.random.choice(self.num_actions_per_dim, p=choice_prob_values[d, :])
            # choice from [0, 1] gets mapped to [-1, 1]
            accels[d] = -1 + 2 * choice
        action = SlideGameAction(accels)

        value_est = value_est[0]  # one element tuple -> float
        return action, value_est


    def prepare_episode(self, episode):
        """ Given an episode of experiences, make some tensors that will be fed
        in for training. The first dimension of the tensors indexes over
        experiences. The tensors rae:
            - state
            - chosen_action
            - reward
            - simple advantage estimator
        """
        ep_len = len(episode.experiences)

        # (ep_len, 3, dim) states
        state = self.states_to_tensor([exp.state for exp in episode.experiences])

        # (ep_len, self.num_actions) one-hot vector of chosen actions
        chosen_action = np.zeros((ep_len, self.dim, self.num_actions_per_dim))
        for i, exp in enumerate(episode.experiences):
            accels = exp.action.accels
            for d in range(self.dim):
                # accel in [-1, 1] gets mapped to [0, 1]
                chosen_action[i, d, (accels[d]+1)/2] = 1.0

        # (ep_len,) vector of rewards
        #reward = episode.compute_cum_future_rewards()
        #reward = episode.compute_cum_discounted_future_rewards(gamma=0.95)
        reward = episode.compute_cum_discounted_future_rewards(gamma=0.0)

        simple_advantage_est = episode.compute_simple_advantage_ests()

        return state, chosen_action, reward, simple_advantage_est


    def do_policy_gradient_update(self, episodes):
        batch_size = sum(len(ep.experiences) for ep in episodes)
        print "batch_size = {}".format(batch_size)

        # make some big tensors out of all the episodes
        prepared_tensors = [self.prepare_episode(ep) for ep in episodes]
        states, chosen_actions, rewards, simple_advantage_ests = zip(*prepared_tensors)
        states = np.concatenate(states, axis=0)
        chosen_actions = np.concatenate(chosen_actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        simple_advantage_ests = np.concatenate(simple_advantage_ests, axis=0)
        print "states =\n{}".format(states)
        print "chosen_actions =\n{}".format(chosen_actions)
        print "rewards =\n{}".format(rewards)
        print "simple_advantage_ests =\n{}".format(simple_advantage_ests)
        assert states.shape == (batch_size, 3, self.dim)
        assert chosen_actions.shape == (batch_size, self.dim, self.num_actions_per_dim)
        assert rewards.shape == (batch_size,)

        prev_linear_coefs = self.sess.run(self.linear_coefs)
        print "prev_linear_coefs = {}".format(prev_linear_coefs)
        #prev_value_coefs = self.sess.run(self.value_coefs)
        #print "prev_value_coefs = {}".format(prev_value_coefs)

        feed_dict = {
            self.state_ph: states,
            self.chosen_action_ph: chosen_actions,
            self.reward_ph: rewards,
#            self.simple_advantage_est_ph: simple_advantage_ests,
            self.policy_learning_rate_ph: 0.5,
            self.value_learning_rate_ph: 0.25,
        }

        [
            _, 
            _,
            choice_probs_value,
            log_choice_probs_value,
            log_chosen_action_prob_value,
#            advantage_thing_value,
            policy_loss_value,
            mean_reward_value,
            value_mse_value,
        ] = self.sess.run([
            self.policy_train_op, 
            self.value_train_op, 
            self.choice_probs, 
            self.log_choice_probs, 
            self.log_chosen_action_prob, 
#            self.advantage_thing,
            self.policy_loss_to_minimize,
            self.mean_reward,
            self.value_mse,
        ], feed_dict=feed_dict)

        print "choice_probs_value = {}".format(choice_probs_value)
        print "log_choice_probs_value = {}".format(log_choice_probs_value)
        print "log_chosen_action_prob_value = {}".format(log_chosen_action_prob_value)
#        print "advantage_thing_value = {}".format(advantage_thing_value)
        print "policy_loss_value = {}".format(policy_loss_value)
        print "mean_reward_value = {}".format(mean_reward_value)
        print "value_mse_value = {}".format(value_mse_value)

        np.set_printoptions(precision=3, suppress=True)
        new_linear_coefs = self.sess.run(self.linear_coefs)
        print "new_linear_coefs = {}".format(new_linear_coefs)
        #new_value_coefs = self.sess.run(self.value_coefs)
        #print "new_value_coefs = {}".format(new_value_coefs)


