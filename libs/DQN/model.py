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

def create_model():
    input = tf.placeholder(tf.float32, 
                           [batch_size, 
                            input_size[0], input_size[1], input_size[2])
    
    
