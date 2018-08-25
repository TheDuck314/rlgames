from GameLoop import play_game

class PeriodicReplayWriter:
    def __init__(self, game_type, agents, period, outdir):
        self.game_type = game_type
        self.agents = agents
        self.period = period
        self.outdir = outdir

    def maybe_write(self, n):
        if n % self.period != 0:
            return

        # nongreedy game
        self.play_and_write("{}/nongreedy/{}.pkl".format(self.outdir, n))
        # greedy game
        for agent in self.agents:
            agent.set_be_greedy(True)
        self.play_and_write("{}/greedy/{}.pkl".format(self.outdir, n))
        for agent in self.agents:
            agent.set_be_greedy(False)


    def play_and_write(self, out_fn):
        play_game(self.game_type(), self.agents).save(out_fn)
        print "saved {}".format(out_fn)

