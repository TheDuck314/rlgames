#!/usr/bin/python

import tensorflow as tf
import random

from GameLoop import play_game
from NPlaceGameQAgent import NPlaceGameQAgent
from NPlaceGame import NPlaceGame
from Replayer import Replayer

def main():
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    agent = NPlaceGameQAgent(sess)
    game_type = NPlaceGame

    sess.run(tf.initialize_all_variables())

    exp_buffer = []
    max_exp_buffer_len = 10000
    minibatch_size = 100

    epsilon = 0.05
    gamma = 0.98

    num_frames_played = 0
    num_frames_trained = 0
    train_frames_per_game_frames = 1.0

    batch = 0
    while True:
        print "Game start"
        agent.set_epsilon(epsilon)
        result = play_game(game_type(), [agent])
        print "game result.episodes[0] = {}".format(result.episodes[0])
        assert len(result.episodes) == 1  # only one-player games supported for now
        game_experiences = result.episodes[0].experiences
        exp_buffer.extend(game_experiences)  # 0th agent's experiences
        num_frames_played += len(game_experiences)
        exp_buffer = exp_buffer[-max_exp_buffer_len:]  # limit experience buffer length

        if len(exp_buffer) >= minibatch_size:
            print "doing Q minibatch"
            while num_frames_trained < train_frames_per_game_frames * num_frames_played:
                print "doing training minibatch because num_frames_trained = {} while num_frames_played = {}".format(num_frames_trained, num_frames_played)
                minibatch = random.sample(exp_buffer, minibatch_size)
                agent.do_q_update(minibatch, gamma)
                num_frames_trained += minibatch_size
            #break
        else:
            print "skipping Q minibatch because len(exp_buffer) is only {}".format(len(exp_buffer))

        batch += 1


if __name__ == "__main__":
    main()
