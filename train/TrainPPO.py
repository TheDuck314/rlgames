#!/usr/bin/python
import sys
from os import path
sys.path.append(path.dirname(path.dirname(__file__)))

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np
import datetime
import os
import time

import Checkpoint
from Annealers import *
from Sampler import Sampler
from PeriodicReplayWriter import PeriodicReplayWriter

from util import GameLoop

from games.NPlaceGame import NPlaceGame
from agents.NPlaceGameTFAgent import NPlaceGameTFAgent
from games.ShootGame import ShootGame
from agents.ShootGameTFAgent import ShootGameTFAgent
from games.SimpleSoccerGame import SimpleSoccerGame, SimpleSoccerDumbAgent
from agents.SimpleSoccerGameTFAgent import SimpleSoccerGameTFAgent

def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])

def train(game_type, agent_type, annealer=None):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    agent = agent_type(sess)
    agents = [agent] * game_type.get_num_agents()
#    agents = [agent, SimpleSoccerDumbAgent()]

    # hyperparameters
    gamma = 0.99
    gae_lambda = 0.95
#    learning_rate = 0.0003
    learning_rate = 0.0001
    epsilon = 0.1  # ppo parameter TODO: fiddle with me
    value_loss_coef = 0.3
    examples_per_iteration = 1000
    minibatch_size = 100  # experiences in each training batch
    epochs = 3
    hyper_string = "ppo_lr{}_ep{}_vc{}_eit{}_mb{}_ep{}".format(learning_rate, epsilon, value_loss_coef, examples_per_iteration, minibatch_size, epochs)

    # set up the training operations in tensorflow
    advantage_ph = tf.placeholder(tf.float32, shape=[None], name='advantage')  # shape: (batch size,)
    old_log_p_chosen_action_ph = tf.placeholder(tf.float32, shape=[None], name='old_log_p_chosen_action')  # shape: (batch size,)
    log_p_chosen_action_op = agent.get_log_p_chosen_action_op()
    p_ratio = tf.exp(log_p_chosen_action_op - old_log_p_chosen_action_ph)
    clipped_p_ratio = tf.clip_by_value(p_ratio, 1.0 - epsilon, 1.0 + epsilon)
    policy_loss = -tf.reduce_sum(tf.minimum(advantage_ph * p_ratio, advantage_ph * clipped_p_ratio))
    # train value function by gradient descent on [(value est) - (cum future reward)] ** 2
    reward_ph = tf.placeholder(tf.float32, shape=[None], name='reward')  # shape: (batch size,)
    value_op = agent.get_value_op()
    value_mse = tf.reduce_sum(tf.square(reward_ph - value_op))
    value_mse_sum = tf.summary.scalar("value_mse", tf.reduce_mean(tf.square(reward_ph - value_op)))
    # put policy and value loss together to get total loss
    total_loss = policy_loss + value_loss_coef * value_mse  # could optionally add an entropy loss to encourage exploration
    learning_rate_ph = tf.placeholder(tf.float32, name="learning_rate")
    train_op = tf.train.AdamOptimizer(learning_rate_ph).minimize(total_loss)

    sess.run(tf.global_variables_initializer())

    exp_buf = []
    rew_buf = []
    adv_buf = []

    prr = PeriodicReplayWriter(game_type=game_type, agents=agents, period=2000, outdir="/home/greg/coding/ML/rlgames/replays")

    merged_sum_op = tf.summary.merge_all()
    log_dir = os.path.join("/home/greg/coding/ML/rlgames/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + hyper_string)
    sum_wri = tf.summary.FileWriter(log_dir, graph=sess.graph, flush_secs=5)

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=0.5)
    ckpt_dir = os.path.join("/home/greg/coding/ML/rlgames", "checkpoints")
    frames_between_ckpts = 5000
    train_frames = Checkpoint.optionally_restore_from_checkpoint(sess, saver, ckpt_dir)
    last_ckpt_frame = train_frames

    start_time = time.time()

    iteration = 0
    while True:
        if annealer:
            annealer.frame(train_frames)

        prr.maybe_write(iteration)

        sampler = Sampler()  # stores the examples we'll use in this iteration

        # play games until we have enough examples to do a round of optimization
        print "iteration {}: playing games...".format(iteration)
        while sampler.num_examples < examples_per_iteration:
            # play a game and remember the experiences and rewards
            # If there's more than one player, the same agent is used for all of them
            # so the agent had better not be something with state like an RNN. Then
            # the experiences of all players are used for training.
            game = game_type()
            result = GameLoop.play_game(game, agents)
            #print result.episodes[0]
            for ep in result.episodes:
                # remember each frame as an example to train on later
                ep_rewards       = ep.compute_cum_discounted_future_rewards(gamma=gamma)
                ep_advs          = ep.compute_generalized_advantage_ests(gamma, gae_lambda)
                ep_log_p_actions = np.array([exp.log_p_action for exp in ep.experiences])
                ep_feed_dict = {
                    reward_ph: ep_rewards,
                    advantage_ph: ep_advs,
                    old_log_p_chosen_action_ph: ep_log_p_actions
                }
                ep_feed_dict.update(agent.make_train_feed_dict(ep.experiences))
                sampler.add_examples(ep_feed_dict)

                # record some stats
                ep_undisc_rewards = ep.compute_cum_discounted_future_rewards(gamma=1.0)
                sum_wri.add_summary(make_summary("disc_rew", ep_rewards[0]), global_step=train_frames)
                sum_wri.add_summary(make_summary("undisc_rew", ep_undisc_rewards[0]), global_step=train_frames)
                sum_wri.add_summary(make_summary("init_value_est", ep.experiences[0].value_est), global_step=train_frames)
                sum_wri.add_summary(make_summary("init_value_mse", (ep.experiences[0].value_est - ep_rewards[0])**2), global_step=train_frames)
            sum_wri.add_summary(make_summary("game_length", len(result.episodes[0].experiences)), global_step=train_frames)
            sum_wri.add_summary(make_summary("total_undisc_rew", sum(sum(exp.reward for exp in ep.experiences) for ep in result.episodes)), global_step=train_frames)

        # do a few epochs of optimization on the examples
        print "iteration {}: starting training...".format(iteration)
        for epoch in range(epochs):
            for mb_i, minibatch_fd in enumerate(sampler.get_minibatches(minibatch_size)):
                #print "begin iteration {} epoch {} minibatch {}".format(iteration, epoch, mb_i)
                minibatch_fd[learning_rate_ph] = learning_rate
                #print "minibatch_fd =\n{}".format(minibatch_fd)
                #print "debug before train step:"
                #agent.print_debug_info()
                [_, sums] = sess.run([train_op, merged_sum_op], feed_dict=minibatch_fd)
                #print "debug after train step:"
                #agent.print_debug_info()
                sum_wri.add_summary(sums, global_step=train_frames)

        iteration += 1
        train_frames += sampler.num_examples
        cur_time = time.time()
        train_frames_per_second = train_frames / (cur_time - start_time)
        print "iteration {}: finished training. total training frames = {}, overall train frames/s = {}".format(iteration, train_frames, train_frames_per_second)

        if train_frames - last_ckpt_frame >= frames_between_ckpts:
            saver.save(sess, os.path.join(ckpt_dir, "model.ckpt"), global_step=train_frames)
            last_ckpt_frame = train_frames


if __name__ == "__main__":
#    train(NPlaceGame, NPlaceGameTFAgent)
    train(SimpleSoccerGame, SimpleSoccerGameTFAgent, SimpleSoccerRewardAnnealer)
#    train(SimpleSoccerGame, SimpleSoccerGameTFAgent)

