#!/usr/bin/python

import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import datetime
import os

import Checkpoint
import GameLoop
from Annealers import *
from PeriodicReplayWriter import PeriodicReplayWriter
from DumbGame import DumbGame
from DumbGameTFAgent import DumbGameTFAgent
from NPlaceGame import NPlaceGame
from NPlaceGameTFAgent import NPlaceGameTFAgent
from ShootGame import ShootGame
from ShootGameTFAgent import ShootGameTFAgent
from SimpleSoccerGame import SimpleSoccerGame, SimpleSoccerDumbAgent
from SimpleSoccerGameTFAgent import SimpleSoccerGameTFAgent

def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])

def train(game_type, agent_type, annealer=None):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    agent = agent_type(sess)
#    agents = [agent] * game_type.get_num_agents()
    agents = [agent, SimpleSoccerDumbAgent()]

    # hyperparameters
    gamma = 0.98
    gae_lambda = 0.95
    #learning_rate = 0.0001
    learning_rate = 0.0003
    minibatch_size = 32  # experiences in each training batch
    #minibatch_size = 1000  # experiences in each training batch
    #value_loss_coef = 0.3
    value_loss_coef = 1.0
    hyper_string = "a2c_lr{}_vc{}_mb{}".format(learning_rate, value_loss_coef, minibatch_size)

    # set up the training operations in tensorflow
    # train policy by gradient descent on -[(log prob chosen action) * (advantage)]
    log_p_chosen_action_op = agent.get_log_p_chosen_action_op()
    advantage_ph = tf.placeholder(tf.float32, shape=[None], name='advantage')  # shape: (batch size,)
    policy_loss = tf.constant(-1.0) * tf.reduce_sum(advantage_ph * log_p_chosen_action_op)
    # train value function by gradient descent on [(value est) - (cum future reward)] ** 2
    reward_ph = tf.placeholder(tf.float32, shape=[None], name='reward')  # shape: (batch size,)
    value_op = agent.get_value_op()
    value_sq_err = tf.square(reward_ph - value_op)
    value_mse_sum = tf.summary.scalar("value_mse", tf.reduce_mean(value_sq_err))
    value_loss = tf.reduce_sum(value_sq_err)
    # train on a combined loss
    total_loss = policy_loss + value_loss_coef * value_loss
    learning_rate_ph = tf.placeholder(tf.float32)
    train_op = tf.train.AdamOptimizer(learning_rate_ph).minimize(total_loss)
    #train_op = tf.train.GradientDescentOptimizer(learning_rate_ph).minimize(total_loss)

    sess.run(tf.global_variables_initializer())

    exp_buf = []
    rew_buf = []
    adv_buf = []

    prr = PeriodicReplayWriter(game_type=game_type, agents=agents, period=50, outdir="/home/greg/coding/ML/rl/replays")

    merged_sum_op = tf.summary.merge_all()
    log_dir = os.path.join("/home/greg/coding/ML/rl/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + hyper_string)
    sum_wri = tf.summary.FileWriter(log_dir, graph=sess.graph, flush_secs=5)

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=0.5)
    ckpt_dir = os.path.join("/home/greg/coding/ML/rl", "checkpoints")
    frames_between_ckpts = 5000
    #train_frames = Checkpoint.optionally_restore_from_checkpoint(sess, saver, ckpt_dir)
    train_frames = 0
    last_ckpt_frame = train_frames

    print "WARNING: only training on 0th agent's experiences"

    ep_num = 0
    while True:
        if annealer:
            annealer.frame(train_frames)

        prr.maybe_write(ep_num)

        # play a game and remember the experiences and rewards
        # If there's more than one player, the same agent is used for all of them
        # so the agent had better not be something with state like an RNN. Then
        # the experiences of all players are used for training.
        game = game_type()
        result = GameLoop.play_game(game, agents)
        #print "len(result.episodes) = {}".format(len(result.episodes))
        for ep in result.episodes[:1]:
            exp_buf.extend(ep.experiences)
            ep_rewards = ep.compute_cum_discounted_future_rewards(gamma=gamma)
            ep_undisc_rewards = ep.compute_cum_discounted_future_rewards(gamma=1.0)
            ep_advs = ep.compute_generalized_advantage_ests(gamma, gae_lambda)
            rew_buf.extend(ep_rewards)
            adv_buf.extend(ep_advs)
            #print "ep_rewards =\n{}".format(ep_rewards)
            #print "ep_advs =\n{}".format(ep_advs)
            #print "exp_buf =\n{}".format(exp_buf)
            #print "rew_buf =\n{}".format(rew_buf)
            #print "adv_buf =\n{}".format(adv_buf)

            sum_wri.add_summary(make_summary("disc_rew", ep_rewards[0]), global_step=train_frames)
            sum_wri.add_summary(make_summary("undisc_rew", ep_undisc_rewards[0]), global_step=train_frames)
            sum_wri.add_summary(make_summary("init_value_est", ep.experiences[0].value_est), global_step=train_frames)
            sum_wri.add_summary(make_summary("init_value_mse", (ep.experiences[0].value_est - ep_rewards[0])**2), global_step=train_frames)
        sum_wri.add_summary(make_summary("game_length", len(result.episodes[0].experiences)), global_step=train_frames)
        sum_wri.add_summary(make_summary("total_undisc_rew", sum(sum(exp.reward for exp in ep.experiences) for ep in result.episodes)), global_step=train_frames)

        # train:
        while len(exp_buf) >= minibatch_size:
            # all this slicing is slow, but whatever
            batch_exps = exp_buf[:minibatch_size]
            batch_rews = rew_buf[:minibatch_size]
            batch_advs = adv_buf[:minibatch_size]
            #print "batch_exps =\n{}".format(batch_exps)
            #print "batch_rews =\n{}".format(batch_rews)
            #print "batch_advs =\n{}".format(batch_advs)
            # create a feed dict that will plug the state and chosen action
            # into the agent's network
            feed_dict = {
                reward_ph: batch_rews,
                advantage_ph: batch_advs,
                learning_rate_ph: learning_rate,
            }
            feed_dict.update(agent.make_train_feed_dict(batch_exps))
            exp_buf = exp_buf[minibatch_size:]  # discard the experiences we used
            rew_buf = rew_buf[minibatch_size:]
            adv_buf = adv_buf[minibatch_size:]
            # do a step of gradient descent
            #print "debug before train:"
            #agent.print_debug_info()
            #print "train feed dict:\n{}".format(feed_dict)
            [_, sums] = sess.run([train_op, merged_sum_op], feed_dict=feed_dict)
            train_frames += minibatch_size
            sum_wri.add_summary(sums, global_step=train_frames)
            #print "debug after train:"
            #agent.print_debug_info()

            if train_frames - last_ckpt_frame >= frames_between_ckpts:
                saver.save(sess, os.path.join(ckpt_dir, "model.ckpt"), global_step=train_frames)
                last_ckpt_frame = train_frames

        ep_num += 1
        #break


if __name__ == "__main__":
    #train(ShootGame, ShootGameTFAgent)
    #train(NPlaceGame, NPlaceGameTFAgent)
#    train(SimpleSoccerGame, SimpleSoccerGameTFAgent, SimpleSoccerRewardAnnealer)
    train(SimpleSoccerGame, SimpleSoccerGameTFAgent)
    #train(DumbGame, DumbGameTFAgent)

