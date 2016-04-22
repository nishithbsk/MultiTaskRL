import tensorflow as tf
import numpy as np

from model_utils import *

def import_model(_model_num, _input_size, _num_actions):
    global model_num
    global input_size
    global num_actions
    model_num = _model_num
    input_size = _input_size
    num_actions = _num_actions    
    return build_model()

def get_Q_values(input_tensor):
    h_conv1 = conv(input_tensor, 8, 8, 32, 4, 4, name='conv1')
    h_pool1 = max_pool(h_conv1, 2, 2, 2, 2, name='pool1')
    h_conv2 = conv(h_pool1, 4, 4, 64, 2, 2, name='conv2')
    h_pool2 = max_pool(h_conv2, 2, 2, 2, 2, name='pool2')
    h_conv3 = conv(h_pool2, 3, 3, 64, 1, 1, name='conv3')
    h_pool3 = max_pool(h_conv3, 2, 2, 2, 2, name='pool3')
    fc1_in = tf.reshape(h_pool3, [-1, 256])
    h_fc1 = fc(fc1_in, 256, name='fc1')
    Q_values = fc(h_fc1, num_actions, relu=False, name='readout_layer')
    return Q_values, h_fc1
    
def build_model():
    input_tensor = tf.placeholder(tf.float32, 
                                  [1, 
                                   input_size[0], input_size[1], input_size[2]])
   
    Q_values, h_fc1 = get_Q_values(input_tensor) 
    return input_tensor, Q_values, h_fc1
