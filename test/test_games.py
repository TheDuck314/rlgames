#!/usr/bin/python
import sys
from os import path
sys.path.append(path.dirname(path.dirname(__file__)))

import pickle

from util import GameLoop, Replayer
from games import DumbGame
from games import NPlaceGame
from games import ShootGame
from games import SimpleSoccerGame
from games import SimpleTeamSoccerGame

def test_DumbGame():
    game = DumbGame.DumbGame()
    agent = DumbGame.DumbAgent()
    result = GameLoop.play_game(game, [agent])
    print result

def test_NPlaceGame():
    game = NPlaceGame.NPlaceGame()
    agent = NPlaceGame.DumbAgent()
    result = GameLoop.play_game(game, [agent])
    print "result =\n{}\n".format(result)
    Replayer.show(result)

def test_ShootGame():
    game = ShootGame.ShootGame()
    agent = ShootGame.DumbAgent()
    result = GameLoop.play_game(game, [agent])
    print "result =\n{}\n".format(result)
    Replayer.show(result)

def test_SimpleSoccerGame():
    game = SimpleSoccerGame.SimpleSoccerGame()
    agents = [SimpleSoccerGame.DumbAgent(), SimpleSoccerGame.DumbAgent()]
    result = GameLoop.play_game(game, agents)
    print "result =\n{}\n".format(result)
    Replayer.show(result)

def test_SimpleTeamSoccerGame():
    game = SimpleTeamSoccerGame.SimpleTeamSoccerGame()
    agents = [SimpleTeamSoccerGame.DumbAgent()] * 4
    result = GameLoop.play_game(game, agents)
    print "result =\n{}\n".format(result)
    Replayer.show(result)

if __name__ == "__main__":
    #test_DumbGame()
    #test_NPlaceGame()
    #test_ShootGame()
    #test_SimpleSoccerGame()
    test_SimpleTeamSoccerGame()
