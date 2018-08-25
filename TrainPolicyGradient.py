#!/usr/bin/python

import tensorflow as tf

from GameLoop import play_game
from DumbGameTFAgent import DumbGameTFAgent
from RangeGameTFAgent import RangeGameTFAgent
from NPlaceGameTFAgent import NPlaceGameTFAgent
from SlideGameTFAgent import SlideGameTFAgent
from ShootGameTFAgent import ShootGameTFAgent
from RangeGame import RangeGame
from NPlaceGame import NPlaceGame
from SlideGame import SlideGame
from ShootGame import ShootGame
from Replayer import Replayer

def main():
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    #agent = DumbGameTFAgent(sess)
    #game_type = DumbGame

    #agent = RangeGameTFAgent(sess)
    #game_type = RangeGame

    #agent = NPlaceGameTFAgent(sess)
    #game_type = NPlaceGame

    #agent = SlideGameTFAgent(sess)
    #game_type = SlideGame

    agent = ShootGameTFAgent(sess)
    game_type = ShootGame

    sess.run(tf.initialize_all_variables())

    #num_games_per_batch = 1000
    num_games_per_batch = 1
    #num_games_per_batch = 1

    batches_between_saved_games = 100

    nongreedy_replay_out_fmt = "/home/greg/coding/ML/rl/replays_nongreedy/train_batch_{batch}.pkl"
    greedy_replay_out_fmt = "/home/greg/coding/ML/rl/replays_greedy/train_batch_{batch}.pkl"

    batch = 0
    while True:
        if batch % batches_between_saved_games == 0:
            # nongreedy game
            out_fn = nongreedy_replay_out_fmt.format(batch=batch)
            play_game(game_type(), [agent]).replay.save(out_fn)
            print "batch {} saved a nongreedy replay to {}".format(batch, out_fn)
            # greedy game
            agent.set_be_greedy(True)
            out_fn = greedy_replay_out_fmt.format(batch=batch)
            play_game(game_type(), [agent]).replay.save(out_fn)
            print "batch {} saved a greedy replay to {}".format(batch, out_fn)
            agent.set_be_greedy(False)

        episodes = []
        for game_i in range(num_games_per_batch):
            result = play_game(game_type(), [agent])
            assert len(result.episodes) == 1  # only one-player games supported for now
            episodes.append(result.episodes[0])  # 0th agent's experiences

        #print "episode batch: {}".format(episodes)

        agent.do_policy_gradient_update(episodes)
        batch += 1
        #break


if __name__ == "__main__":
    main()
