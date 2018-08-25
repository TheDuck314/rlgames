from GameLoop import GameResult
from games.NPlaceGame import *
from games.SlideGame import *
from games.ShootGame import *
from games.SimpleSoccerGame import *

from graphics import *

class Replayer:
    @staticmethod
    def show(game_result, fn=None):
        assert "true_states" in game_result.__dict__
        assert len(game_result.true_states) > 0

        # peek at the type of the first state to decide how to show the replay
        first_state = game_result.true_states[0]
        if isinstance(first_state, NPlaceGameState):
            Replayer.show_NPlaceGame(game_result, fn=fn)
        if isinstance(first_state, SlideGameState):
            Replayer.show_SlideGame(game_result, fn=fn)
        if isinstance(first_state, ShootGameState):
            Replayer.show_ShootGame(game_result, fn=fn)
        if isinstance(first_state, SimpleSoccerState):
            Replayer.show_SimpleSoccer(game_result, fn=fn)
        else:
            raise Exception("Replayer doesn't support state_type {}".format(type(first_state)))

    @staticmethod
    def load_and_show(fn):
        game_result = GameResult.load(fn)
        Replayer.show(game_result)


    @staticmethod
    def show_NPlaceGame(game_result, fn=None):
        loc_width = 50
        win_height = 200

        num_locs = NPlaceGame.num_locations()
        win_width = loc_width * num_locs

        win_name = "Replay"
        if fn:
            win_name += " " + fn
        win = GraphWin(win_name, win_width, win_height, autoflush=False)

        min_reward = min(NPlaceGame.loc_rewards)
        max_reward = max(NPlaceGame.loc_rewards)
        def reward_to_color(reward):
            x = int((reward - min_reward) * 255 / (max_reward - min_reward))
            return color_rgb(255 - x, x, 0)

        # draw the map
        for loc in range(num_locs):
            left_x = loc * loc_width
            right_x = left_x + loc_width
            center_y = win_height / 2
            top_y = center_y - loc_width / 2
            bottom_y = center_y + loc_width / 2
            sq = Rectangle(Point(left_x, top_y), Point(right_x, bottom_y))
            sq.setWidth(10)
            #sq.setFill("green")
            sq.setFill(reward_to_color(NPlaceGame.loc_rewards[loc]))
            sq.draw(win)

        update()

        player_circ = Circle(Point(0,0), 0.4 * loc_width)
        player_circ.setFill("blue")
        player_circ.draw(win)
        frame_num_text = Text(Point(50, 50), "frame: 0/{}".format(len(game_result.true_states)-1))
        frame_num_text.draw(win)

        for frame_num, state in enumerate(game_result.true_states):
            frame_num_text.setText("frame: {}/{}".format(frame_num, len(game_result.true_states)-1))
            new_x = (loc_width / 2) + state.location * loc_width
            player_circ.moveCenter(new_x, win_height/2)
            update(10.0)

        game_over_text = Text(Point(350, 50), "GAME OVER -- closing")
        game_over_text.setFill("red")
        game_over_text.draw(win)
        update()

        #win.getMouse() # Pause to view result
        time.sleep(0.5)  # Pause to view result
        win.close()    # Close window when done


    @staticmethod
    def show_SlideGame(replay, fn=None):
        scale = 120
        win_radius = 4

        win_width = scale * win_radius * 2
        win_height = win_width

        win_name = "Replay"
        if fn:
            win_name += " " + fn
        win = GraphWin(win_name, win_width, win_height, autoflush=False)
        win.setCoords(-win_radius, -win_radius, win_radius, win_radius)

        player_circ = Circle(Point(0,0), 0.1)
        player_circ.setFill("blue")
        player_circ.draw(win)
        vel_line = Line(Point(0, 0), Point(0, 0))
        dest_circ = Circle(Point(0,0), 0.1)
        dest_circ.setFill("green")
        dest_circ.draw(win)
        reward_circ1 = Circle(Point(0,0), SlideGame.reward_decay_length)
        reward_circ1.draw(win)
        reward_circ2 = Circle(Point(0,0), 2 * SlideGame.reward_decay_length)
        reward_circ2.draw(win)
        frame_num_text = Text(Point(-3, -3), "frame: 0/{}".format(len(game_result.true_states)-1))
        frame_num_text.draw(win)

        for frame_num, state in enumerate(game_result.true_states):
            frame_num_text.setText("frame: {}/{}".format(frame_num, len(game_result.true_states)-1))

            player_circ.moveCenter(state.position[0], state.position[1])
            
            dest_circ.moveCenter(state.dest[0], state.dest[1])
            reward_circ1.moveCenter(state.dest[0], state.dest[1])
            reward_circ2.moveCenter(state.dest[0], state.dest[1])

            vel_line.undraw()
            vel_line = Line(Point(state.position[0], state.position[1]),
                            Point(state.position[0] + 1.0 * state.velocity[0], 
                                  state.position[1] + 1.0 * state.velocity[1]))
            vel_line.setWidth(5)
            vel_line.setFill("red")
            vel_line.draw(win)

            update(30.0)

        game_over_text = Text(Point(350, 50), "GAME OVER -- closing")
        game_over_text.setFill("red")
        game_over_text.draw(win)
        update()

        #win.getMouse() # Pause to view result
        time.sleep(1.0)  # Pause to view result
        win.close()    # Close window when done


    @staticmethod
    def show_ShootGame(game_result, fn=None):
        scale = 100
        win_radius = 5

        win_width = scale * win_radius * 2
        win_height = win_width

        win_name = "Replay"
        if fn:
            win_name += " " + fn
        win = GraphWin(win_name, win_width, win_height, autoflush=False)
        win.setCoords(-win_radius, -win_radius, win_radius, win_radius)

        player_circ = Circle(Point(0,0), 0.3)
        player_circ.setFill("blue")
        player_circ.draw(win)
        #vel_line = Line(Point(0, 0), Point(0, 0))
        puck_circ = Circle(Point(0,0), 0.15)
        puck_circ.setFill("red")
        puck_circ.draw(win)
        goal_circ = Circle(Point(0,0), ShootGame.goal_radius)
        goal_circ.draw(win)
        goal_ray_1 = Line(Point(0, 0), Point(2 * win_radius * math.cos(ShootGame.goal_opening_half_angle), 
                                             2 * win_radius * math.sin(ShootGame.goal_opening_half_angle)))
        goal_ray_2 = Line(Point(0, 0), Point(2 * win_radius * math.cos(ShootGame.goal_opening_half_angle), 
                                            -2 * win_radius * math.sin(ShootGame.goal_opening_half_angle)))
        goal_ray_1.setFill("green")
        goal_ray_2.setFill("green")
        goal_ray_1.setWidth(5)
        goal_ray_2.setWidth(5)
        goal_ray_1.draw(win)
        goal_ray_2.draw(win)
        frame_num_text = Text(Point(-3, -3), "frame: 0/{}".format(len(game_result.true_states)-1))
        frame_num_text.draw(win)
        value_text = Text(Point(-3, -2), "value")
        value_text.draw(win)

        for frame_num, state in enumerate(game_result.true_states):
            frame_num_text.setText("frame: {}/{}".format(frame_num, len(game_result.true_states)-1))
            if frame_num < len(game_result.episodes[0].experiences):
                value_text.setText("value: {}".format(game_result.episodes[0].experiences[frame_num].value_est))

            player_circ.moveCenter(state.agent.position[0], state.agent.position[1])
            puck_circ.moveCenter(state.puck.position[0], state.puck.position[1])

            update(10.0)

        game_over_text = Text(Point(350, 50), "GAME OVER -- closing")
        game_over_text.setFill("red")
        game_over_text.draw(win)
        update()

        #win.getMouse() # Pause to view result
        time.sleep(1.0)  # Pause to view result
        win.close()    # Close window when done


    @staticmethod
    def show_SimpleSoccer(game_result, fn=None):
        scale = 150
        win_radius = 3.5

        win_width = scale * win_radius * 2
        win_height = win_width * 0.8

        win_name = "Replay"
        if fn:
            win_name += " " + fn
        win = GraphWin(win_name, win_width, win_height, autoflush=False)
        win.setCoords(-win_radius, -win_radius*0.95, win_radius, win_radius*0.65)

        max_x = SimpleSoccerGame.max_x
        max_y = SimpleSoccerGame.max_y

        arena_lines = [
            (-max_x, -max_y, -max_x,  max_y, "blue"),
            ( max_x, -max_y,  max_x,  max_y, "blue"),
            (-max_x, -max_y,  max_x, -max_y, "black"),
            (-max_x,  max_y,  max_x,  max_y, "black"),
            (   0.0, -max_y,    0.0,  max_y, "black"),
        ]

        for (x1, y1, x2, y2, color) in arena_lines:
            arena_line = Line(Point(x1, y1), Point(x2, y2))
            arena_line.setFill(color)
            arena_line.draw(win)

        agent0_circ = Circle(Point(0,0), 0.3)
        agent0_circ.setFill("red")
        agent0_circ.draw(win)

        agent1_circ = Circle(Point(0,0), 0.3)
        agent1_circ.setFill("green")
        agent1_circ.draw(win)

        puck_circ = Circle(Point(0,0), 0.15)
        puck_circ.setFill("blue")
        puck_circ.draw(win)

        frame_num_text = Text(Point(0, -3.1), "frame: 0/{}".format(len(game_result.true_states)-1))
        frame_num_text.draw(win)
        value0_text = Text(Point(0, -2.5), "value0")
        value0_text.draw(win)
        value1_text = Text(Point(0, -2.8), "value1")
        value1_text.draw(win)

        for frame_num, state in enumerate(game_result.true_states):
            frame_num_text.setText("frame: {}/{}".format(frame_num, len(game_result.true_states)-1))
            if frame_num < len(game_result.episodes[0].experiences):
                value0_text.setText("value0: {}".format(game_result.episodes[0].experiences[frame_num].value_est))
                value1_text.setText("value1: {}".format(game_result.episodes[1].experiences[frame_num].value_est))

            agent0_circ.moveCenter(state.agent0.pos[0], state.agent0.pos[1])
            agent1_circ.moveCenter(state.agent1.pos[0], state.agent1.pos[1])
            puck_circ.moveCenter(state.puck.pos[0], state.puck.pos[1])

            #update(1.0)
            update(15.0)

        #game_over_text = Text(Point(350, 50), "GAME OVER -- closing")
        #game_over_text.setFill("red")
        #game_over_text.draw(win)
        #update()

        #win.getMouse() # Pause to view result
        time.sleep(1.0)  # Pause to view result
        win.close()    # Close window when done
