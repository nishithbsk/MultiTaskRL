import sys
import cv2
import numpy as np

from ple import PLE

fps = 30
frame_skip = 1
num_steps = 1
force_fps = True
display_screen = True
reward_values = {} # for customized reward values

def map_game_to_PLE(game_name):
    print game_name
    game_class = None
    if game_name == 'catcher':
        from ple.games.catcher import Catcher
        game_class = Catcher()
    elif game_name == 'flappybird':
        from ple.games.flappybird import FlappyBird
        game_class = FlappyBird()
    elif game_name == 'pixelcopter':
        from ple.games.pixelcopter import Pixelcopter
        game_class = Pixelcopter()
    elif game_name == 'pong':
        from ple.games.pong import Pong
        game_class = Pong()
    elif game_name == 'puckworld':
        from ple.games.puckworld import PuckWorld
        game_class = PuckWorld()
    elif game_name == 'raycastmaze':
        from ple.games.raycastmaze import RaycastMaze
        game_class = RaycastMaze()
    elif game_name == 'snake':
        from ple.games.snake import Snake
        game_class = Snake()
    elif game_name == 'waterworld':
        from ple.games.waterworld import WaterWorld
        game_class = WaterWorld()
    else:
        print "Not a valid game name. Exiting..."
        sys.exit(1)
    return PLE(game_class, fps=fps, frame_skip=frame_skip, 
               num_steps=num_steps, force_fps=force_fps, 
               display_screen=display_screen)
 
def create_game(game_file, is_ale, is_ple):
    if is_ale and is_ple:
        print "Game file cannot contain both ALE and PLE games"
        sys.exit(1)
    if not is_ale and not is_ple:
        print "Please specify a game learning environment"
        sys.exit(1)

    games, names = [], []
    with open(game_file, 'r') as f:
        game_names = f.readlines()
        for game_name in game_names:
            game_name = game_name.lower()[:-1]
            if is_ple:
                games.append(map_game_to_PLE(game_name))
            names.append(game_name)
    return games, names

def step(game, action_index, stacked_old_state, dummy_try=False):
    reward = game.act(game.getActionSet()[action_index])

    new_state = game.getScreenGrayscale()
    new_state = cv2.resize(new_state, (80, 80))
    _, new_state = cv2.threshold(new_state, 1, 255, cv2.THRESH_BINARY)
    
    if dummy_try:
        stacked_new_state = np.stack((new_state,
                                      new_state,
                                      new_state,
                                      new_state), axis = 2)
    else:
        new_state = np.reshape(new_state, (80, 80, 1))
        stacked_new_state = np.append(new_state, 
                                      stacked_old_state[:, :, :3], axis=2)

    is_terminal = game.game_over()
    if is_terminal:
        game.reset_game()
    
    return stacked_new_state, stacked_old_state, reward, is_terminal

