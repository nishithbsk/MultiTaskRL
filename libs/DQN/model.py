import tensorflow as tf
import numpy as np

from model_utils import *

def import_model(model_num, batch_size, input_size, num_actions):
    global model_num
    global batch_size
    global input_size
    global num_actions
    model_num = model_num
    batch_size = batch_size
    input_size = input_size
    num_actions = num_actions
    
    return create_model()

def get_readout(input):
    h_conv1 = conv(input, 8, 8, 32, 4, 4, name='conv1')
    h_pool1 = max_pool(h_conv1, 2, 2, 2, 2, name='pool1')
    h_conv2 = conv(h_pool1, 4, 4, 64, 2, 2, name='conv2')
    h_pool2 = max_pool(h_conv2, 2, 2, 2, 2, name='pool2')
    h_conv3 = conv(h_pool2, 3, 3, 64, 1, 1, name='conv3')
    h_pool3 = max_pool(h_conv3, 2, 2, 2, 2, name='pool3')
    fc1_in = tf.reshape(h_pool3, [batch_size, 256, 1])
    h_fc1 = fc(fc1_in, 256, name='fc1')
    readout = fc(h_fc1, 3, relu=False, name='readout_layer')
    return readout, h_fc1
    
def create_model():
    input = tf.placeholder(tf.float32, 
                           [batch_size, 
                            input_size[0], input_size[1], input_size[2])
   
    readout, h_fc1 = get_readout(input) 
    return input, readout, h_fc1
