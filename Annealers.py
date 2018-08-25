from SimpleSoccerGame import SimpleSoccerGame

class SimpleSoccerRewardAnnealer:
    @staticmethod
    def frame(f):
        # anneal loss_reward from 0 to -1
        # starting it at zero makes the agent more likely to learn to
        # pick up and shoot the ball, since own goals aren't penalized
        progress = max(0.0, min(1.0, f / 2e6))
        if progress < 0.2:
            SimpleSoccerGame.loss_reward = 0.0
            SimpleSoccerGame.max_frames = 200
        else:
            SimpleSoccerGame.loss_reward = -(progress - 0.2) / 0.8
            SimpleSoccerGame.max_frames = int(200 + 1800 * (progress - 0.2) / 0.8)
        print "SimpleSoccerRewardAnnealer progress = {}  loss_reward = {}".format(progress, SimpleSoccerGame.loss_reward)


