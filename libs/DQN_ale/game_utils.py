import sys
import cv2
import numpy as np

from ale_python_interface import ALEInterface

def setup_display(game):
    USE_SDL = True
    if USE_SDL:
        if sys.platform == 'darwin':
            import pygame
            pygame.init()
            game.setBool('sound', False) # Sound doesn't work on OSX
        elif sys.platform.startswith('linux'):
            game.setBool('sound', True)
        game.setBool('display_screen', True)

def map_game_to_ALE(game_name, interactive):
    game_path = '/cvgl/u/nishith/MultiTaskRL/libs/DQN_ale/roms/' \
                + game_name + '.bin'
    print game_path
    game = ALEInterface()
    if interactive:
        setup_display(game)
    game.loadROM(game_path)
    return game

def create_game(game_file, interactive):
    games, names, masks = [], [], []
    with open(game_file, 'r') as f:
        game_names = f.readlines()
        for name in game_names:
            name = name.lower()[:-1]
            names.append(name)

            game = map_game_to_ALE(name, interactive)
            games.append(game)

            valid_actions = game.getMinimalActionSet()
            all_actions = game.getLegalActionSet()
            valid_idx = np.where(np.in1d(all_actions, valid_actions))
            mask = np.zeros_like(all_actions)
            mask[valid_idx] = 1
            masks.append(mask)

    return games, names, masks

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

