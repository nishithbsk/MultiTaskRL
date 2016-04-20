from ple import PLE

fps = 30
frame_skip = 2
num_steps = 1
reward = 0.0
nb_frames = 15000
force_fps = True
display_screen = True

def map_int_to_game(game_index):
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
    return PLE(gameClass, fps=fps, frame_skip=frame_skip, num_steps=num_steps, 
               force_fps=force_fps, display_screen=display_screen)
 
def create_game(game_indices):
    games = []
    for game_index in game_indices:
        games.append(map_int_to_game(game_index))
    return games

