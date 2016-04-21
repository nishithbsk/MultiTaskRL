from ple import PLE

fps = 30
frame_skip = 2
num_steps = 1
reward = 0.0
nb_frames = 15000
force_fps = True
display_screen = True

def map_int_to_game(game_index):
    game_index = int(game_index)
    game_class = None
    if game_index == 1:
        from ple.games.catcher import Catcher
        game_class = Catcher()
    elif game_index == 2:
        from ple.games.flappybird import FlappyBird
        game_class = FlappyBird()
    elif game_index == 3:
        from ple.games.pixelcopter import Pixelcopter
        game_class = Pixelcopter()
    elif game_index == 4:
        from ple.games.pong import Pong
        game_class = Pong()
    elif game_index == 5:
        from ple.games.puckworld import PuckWorld
        game_class = PuckWorld()
    elif game_index == 6:
        from ple.games.raycastmaze import RaycastMaze
        game_class = RaycastMaze()
    elif game_index == 7:
        from ple.games.snake import Snake
        game_class = Snake()
    elif game_index == 8:
        from ple.games.waterworld import WaterWorld
        game_class = WaterWorld()
    return PLE(game_class, fps=fps, frame_skip=frame_skip, \
               num_steps=num_steps, force_fps=force_fps, \
               display_screen=display_screen)
 
def create_game(game_indices):
    games = []
    for game_index in game_indices:
        games.append(map_int_to_game(game_index))
    return games

def step(game, action, stacked_old_state, dummy_try=False):
    reward = game.act(action)
      
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

