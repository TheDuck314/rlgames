#!/usr/bin/python

import GameLoop
from DumbGame import *
from RangeGame import *
from NPlaceGame import *
from SlideGame import *
from ShootGame import *
from SimpleSoccerGame import *
from Replayer import Replayer
import pickle

def test_DumbGame():
    game = DumbGame()
    agent = DumbGameDumbAgent()
    result = GameLoop.play_game(game, [agent])
    print result

def test_RangeGame():
    game = RangeGame()
    agent = RangeGameDumbAgent()
    experiences = GameLoop.play_game(game, [agent])
    print experiences

def test_NPlaceGame():
    game = NPlaceGame()
    agent = NPlaceGameDumbAgent()
    result = GameLoop.play_game(game, [agent])
    print "result.experiences =\n{}\n".format(result.episodes[0].experiences)
    print "result.replay =\n{}".format(result.replay)
    out_fn = "/home/greg/coding/ML/rl/replays/test.pkl"
    result.replay.save(out_fn)
    print "wrote replay to {}".format(out_fn)
    Replayer.show(result.replay)

def test_SlideGame():
    game = SlideGame()
    agent = SlideGameDumbAgent()
    result = GameLoop.play_game(game, [agent])
    print "result.experiences =\n{}\n".format(result.episodes[0].experiences)
    print "result.replay =\n{}".format(result.replay)
    #out_fn = "/home/greg/coding/ML/rl/replays/test.pkl"
    #result.replay.save(out_fn)
    #print "wrote replay to {}".format(out_fn)
    Replayer.show(result.replay)

def test_ShootGame():
    game = ShootGame()
    agent = ShootGameDumbAgent()
    result = GameLoop.play_game(game, [agent])
    print "result.experiences =\n{}\n".format(result.episodes[0].experiences)
    print "result.replay =\n{}".format(result.replay)
    #out_fn = "/home/greg/coding/ML/rl/replays/test.pkl"
    #result.replay.save(out_fn)
    #print "wrote replay to {}".format(out_fn)
    Replayer.show(result.replay)

def test_SimpleSoccerGame():
    game = SimpleSoccerGame()
    agents = [SimpleSoccerDumbAgent(), SimpleSoccerDumbAgent()]
    result = GameLoop.play_game(game, agents)
    print "result.experiences =\n{}\n".format(result.episodes[0].experiences)
    print "result.true_states =\n{}".format(result.true_states)
    #out_fn = "/home/greg/coding/ML/rl/replays/test.pkl"
    #result.replay.save(out_fn)
    #print "wrote replay to {}".format(out_fn)
    Replayer.show(result)

if __name__ == "__main__":
    test_DumbGame()
    #test_RangeGame()
    #test_NPlaceGame()
    #test_SlideGame()
    #test_ShootGame()
    #test_SimpleSoccerGame()
