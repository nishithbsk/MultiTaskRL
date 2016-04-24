import tensorflow as tf
import numpy as np

# Based off Ethereon's implementation of Caffe-Tensorflow:
# https://github.com/ethereon/caffe-tensorflow/blob/master/kaffe/tensorflow/network.py

DEFAULT_PADDING = 'SAME'

def make_var(name, shape):
    return tf.get_variable(name, shape)

def validate_padding(padding):
    assert padding in ('SAME', 'VALID')

def conv(input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING):
    validate_padding(padding)
    c_i = input.get_shape()[-1]
    with tf.variable_scope(name) as scope:
        kernel = make_var('weights', shape=[k_h, k_w, c_i, c_o])
        biases = make_var('biases', [c_o])
        conv = tf.nn.conv2d(input, kernel, strides = [1, s_h, s_w, 1], padding=padding)
        if relu:
            return tf.nn.relu(conv + biases)
        else:
            return conv + biases

def relu(input, name):
    return tf.nn.relu(input, name=name)

def max_pool(input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
    validate_padding(padding)
    return tf.nn.max_pool(input,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)

def avg_pool(input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
    validate_padding(padding)
    return tf.nn.avg_pool(input,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)

def lrn(input, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(input,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias,
                                              name=name)

def concat(inputs, axis, name):
    return tf.concat(concat_dim=axis, values=inputs, name=name)

def fc(input, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:
        input_shape = input.get_shape()
        if input_shape.ndims==4:
            dim = 1
            for d in input_shape[1:].as_list():
                dim *= d
            feed_in = tf.reshape(input, [int(input_shape[0]), dim])
        else:
            feed_in, dim = (input, int(input_shape[-1]))
        weights = make_var('weights', shape=[dim, num_out])
        biases = make_var('biases', [num_out])
        op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
        fc = op(feed_in, weights, biases, name=scope.name)
        return fc

def softmax(input, name):
    return tf.nn.softmax(input, name)

def dropout(input, keep_prob, name):
    return tf.nn.dropout(input, keep_prob, name=name)
