import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import argparse

from collections import deque
from ple import PLE
from model import *

# Parse arguments
parser = argparse.ArgumentParser(description='Train a Deep-Q-Network')
parser.add_argument('-g', '--game', help='Name of game')
parser.add_argument('-na', '--num_actions', type=int, help='Number of actions')
args = parser.parse_args()

# Constants
game_name = args.game # the name of the game being played for log files
num_actions = args.num_actions # number of valid actions
batch_size = 32 # size of minibatch
gamma = 0.99 # decay rate of past observations
observe = 500 # timesteps to observe before training
explore = 500 # frames over which to anneal epsilon
initial_epsilon = 1.0 # starting value of epsilon
final_epsilon = 0.05 # final value of epsilon
replay_memory = 590000 # number of previous transitions to remember
K = 1 # only select an action every Kth frame, repeat prev for others

# Pick an action given state
def pickAction(sess, state, readout):
    return sess.run(readout, feed_dict={input = state})

def train(sess, s, readout, h_fc1):
    

def main():
    # launch a session
    sess = tf.InteractiveSession()
    # define network
    s, readout, h_fc1 = build_model()
    # train network
    train(sess, s, readout, h_fc1)

if __name__ == '__main__":
    main()
