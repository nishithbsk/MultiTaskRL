import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import argparse

from collections import deque
from ple import PLE
from model import *
from game_utils import *

# Parse arguments
parser = argparse.ArgumentParser(description='Train a Deep-Q-Network')
parser.add_argument('-gi', '--game_indices', type=tuple, help='List of game indices')
parser.add_argument('-na', '--num_actions', type=int, help='Number of actions')
args = parser.parse_args()

# Constants
game_indices = args.game_indices # the name of the game being played for log files
num_actions = args.num_actions # number of valid actions
batch_size = 32 # size of minibatch
gamma = 0.99 # decay rate of past observations
observe = 500 # timesteps to observe before training
explore = 500 # frames over which to anneal epsilon
initial_epsilon = 1.0 # starting value of epsilon
final_epsilon = 0.05 # final value of epsilon
replay_memory = 590000 # number of previous transitions to remember
K = 1 # only select an action every Kth frame, repeat prev for others

# Forward pass to obtain Q-values for every action
def get_Q_values(sess, state, readout):
    return sess.run(readout, feed_dict={input = state})

def calculate_loss(readout):
    a = tf.placeholder("float", [None, num_actions])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.mul(readout, a), reduction_indices = 1)
    loss = tf.reduce_mean(tf.square(y - readout_action))
    return loss

def train(sess, s, readout, h_fc1):
    loss = calculate_loss(readout)
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

def setup_game():
    global game
    game = create_game(game_indices)[0]

def main():
    # Launch a session
    sess = tf.InteractiveSession()
    # Set up game
    setup_game()
    # Define network
    s, readout, h_fc1 = build_model()
    # Train network
    train(sess, s, readout, h_fc1)

if __name__ == '__main__":
    main()
