import tensorflow as tf
import numpy as np
import cv2
import random
import os
import sys
import argparse

from collections import deque
from ple import PLE
from model import *
from game_utils import create_game, step

# Parse arguments
parser = argparse.ArgumentParser(description='Train a Deep-Q-Network')
parser.add_argument('-gi', '--game_indices', type=tuple, help='List of game indices')
parser.add_argument('-na', '--num_actions', type=int, help='Number of actions')
args = parser.parse_args()

# Constants
game_indices = args.game_indices # the name of the game being played for log files
num_actions = args.num_actions # number of valid actions
input_size = (80, 80, 4)
batch_size = 32 # size of minibatch
gamma = 0.99 # decay rate of past observations
observe = 500 # timesteps to observe before training
explore = 500 # frames over which to anneal epsilon
initial_epsilon = 1.0 # starting value of epsilon
final_epsilon = 0.05 # final value of epsilon
replay_memory = 590000 # number of previous transitions to remember
frames_per_action = 1 # only select an action every Kth frame, repeat prev for others

# Given Q values for every possible valid action
# choose action.
def pick_action(Q_values_t, epsilon, t):
    action_onehot_t = np.zeros((num_actions))
    action_index = 0
    if t % frames_per_action == 0:
        if random.random() <= epsilon:
            action_onehot_t[random.randrange(num_actions)] = 1
        else:
            action_onehot_t[np.argmax(Q_values_t)] = 1
    else:
        action_onehot_t[game.NOOP] = 1
    return action_onehot_t

# Forward pass to obtain Q-values for every action
def update_Q_values(sess, state, Q_values):
    return sess.run(Q_values, feed_dict={input: state})

def calculate_loss(Q_values):
    action_onehot = tf.placeholder("float", [None, num_actions])
    y = tf.placeholder("float", [None])
    current_Q = tf.reduce_sum(tf.mul(Q_values, action_onehot), 
                              reduction_indices = 1)
    loss = tf.reduce_mean(tf.square(y - current_Q))
    return loss

def train(sess, state, Q_values, h_fc1):
    loss = calculate_loss(Q_values)
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    D = deque()
    
    # Obtain first frame
    s_0, _, r_0, is_terminal = step(game, game.NOOP,
                             stacked_old_state=None, dummy_try=True)

    sess.run(tf.initialize_all_variables())

    epsilon = initial_epsilon
    t = 0
    while True:
        Q_values_t = update_Q_values(sess, [state], Q_values)[0]
        action_onehot_t = pick_action(Q_values_t, epsilon, t)
        
        # scale down epsilon
        if epsilon > final_epsilon and t > observe:
            epsilon -= (initial_epsilon - final_epsilon) / explore

        s_t, s, r_t, is_terminal = step(game, np.argmax(action_onehot_t),
                                     stacked_old_state=s)
        
        # store the transition in D
        D.append((s, action_onehot_t, r_t, s_t, is_terminal))

        if len(D) > replay_memory:
            D.popleft()

        # only train if done observing
        if t > observe:
            # sample a minibatch to train on
            minibatch = random.sample(D, batch_size)

            # get the batch variables
            s_batch = [d[0] for d in minibatch]
            action_onehot_batch = [d[1] for d in minibatch]
            r_t_batch = [d[2] for d in minibatch]
            s_t_batch = [d[3] for d in minibatch]

            y_batch = []
            Q_values_t_batch = update_Q_values(sess, s_t_batch, Q_values)
            for i in range(len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_t_batch[i])
                else:
                    y_batch.append(r_t_batch[i] + \
                                   gamma * np.max(Q_values_t_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                action_onehot : action_onehot_batch,
                state : s_batch}
            )

        # update the old values
        s = s_t
        t += 1

def setup_game():
    global game
    game = create_game(game_indices)[0]

def main():
    # Launch a session
    sess = tf.InteractiveSession()
    # Set up game
    setup_game()
    # Define network
    state, Q_values, h_fc1 = import_model(1, input_size, num_actions)
    # Train network
    train(sess, state, Q_values, h_fc1)

if __name__ == "__main__":
    main()
