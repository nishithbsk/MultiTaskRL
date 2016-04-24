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
args = parser.parse_args()

# Constants
game_indices = args.game_indices # the name of the game being played for log files
num_actions = None # number of valid actions
input_size = (80, 80, 4)
batch_size = 32 # size of minibatch
gamma = 0.99 # decay rate of past observations
observe = 50 # timesteps to observe before training
explore = 500 # frames over which to anneal epsilon
initial_epsilon = 1.0 # starting value of epsilon
final_epsilon = 0.05 # final value of epsilon
replay_memory = 590000 # number of previous transitions to remember
frames_per_action = 1 # only select an action every Kth frame

def get_status(t):
    status = ""
    if t <= observe:
        status = "observe"
    elif t > observe and t <= observe + explore:
        status = "explore"
    else:
        status = "train"
    return status

# Given Q values for every possible valid action, choose action
def pick_action(Q_values_t, epsilon, t):
    a_t = np.zeros((num_actions))
    action_index = 0
    if t % frames_per_action == 0:
        if random.random() <= epsilon:
            a_t[random.randrange(num_actions)] = 1
        else:
            a_t[np.argmax(Q_values_t)] = 1
    else:
        a_t[-1] = 1
    return a_t

def calculate_target(Q_values, state, minibatch):
    target_batch, s_batch, a_t_batch = [], [], []
    for d in minibatch:
        s_d, a_t_d, r_t_d, s_t_d, terminal_d = d
        Q_values_t_d = Q_values.eval(feed_dict={state: [s_t_d]})[0]
        if terminal_d:
            target_batch.append(r_t_d)
        else:
            target_batch.append(r_t_d + gamma*np.max(Q_values_t_d))
        s_batch.append(s_d)
        a_t_batch.append(a_t_d)

    return target_batch, s_batch, a_t_batch

def calculate_loss(Q_values):
    action = tf.placeholder("float", [None, num_actions])
    target = tf.placeholder("float", [None])
    current_Q = tf.reduce_sum(tf.mul(Q_values, action), reduction_indices = 1)
    loss = tf.reduce_mean(tf.square(target - current_Q))
    return loss, action, target

def train(sess, state, Q_values, h_fc1):
    loss, action, target = calculate_loss(Q_values)
    train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)

    D = deque()
    
    # Obtain first frame
    s, _, r, is_terminal = step(game, -1,
                                stacked_old_state=None, dummy_try=True)
    
    sess.run(tf.initialize_all_variables())

    epsilon = initial_epsilon
    t = 0
    while True:
        Q_values_t = Q_values.eval(feed_dict={state: [s]})[0]
        a_t = pick_action(Q_values_t, epsilon, t)
        
        # scale down epsilon
        if epsilon > final_epsilon and t > observe:
            epsilon -= (initial_epsilon - final_epsilon) / explore

        s_t, s, r_t, is_terminal = step(game, np.argmax(a_t),
                                        stacked_old_state=s)
        
        # store the transition in D
        D.append((s, a_t, r_t, s_t, is_terminal))

        if len(D) > replay_memory:
            D.popleft()

        # only train if done observing
        if t > observe:
            # sample a minibatch to train on
            minibatch = random.sample(D, batch_size)

            target_batch, s_batch, a_t_batch = \
                                calculate_target(Q_values, state, minibatch)

            # perform gradient step
            sess.run(train_step, 
                     feed_dict = {target : target_batch,
                                  action : a_t_batch,
                                  state : s_batch})
            
        # print info
        status = get_status(t)
        print "timestep:", t, "status:", status, "action:", np.argmax(a_t),\
              "reward:", r_t, "max_Q:", np.max(Q_values_t), "epsilon:", epsilon

        # update the old values
        s = s_t
        t += 1

def setup_game():
    global game
    game = create_game(game_indices)[0]
    game.init()
    global num_actions
    num_actions = len(game.getActionSet())
    print num_actions

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
