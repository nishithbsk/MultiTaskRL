import sys
import cv2
import numpy as np

from ale_python_interface import ALEInterface

def map_game_to_ALE(game_name):
    game_path = '/cvgl/u/nishith/MultiTaskRL/libs/DQN_ale/roms/' \
                + game_name + '.bin'
    print game_path
    game = ALEInterface()
    game.loadROM(game_path)
    return game

def create_game(game_file):
    games, names = [], []
    with open(game_file, 'r') as f:
        game_names = f.readlines()
        for game_name in game_names:
            game_name = game_name.lower()[:-1]
            games.append(map_game_to_ALE(game_name))
            names.append(game_name)
    return games, names

def step(game, action_index, stacked_old_state, dummy_try=False):
    reward = game.act(game.getLegalActionSet()[action_index])

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

