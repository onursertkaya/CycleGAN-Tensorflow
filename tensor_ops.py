#!/usr/bin/python3

import tensorflow as tf

# notes
# truncated_normal_initializer does not allow values larger than 2*stddev
# other than that, it is same as tf.random_normal_initializer

# reflection padding taken from
# https://github.com/lengstrom/fast-style-transfer/issues/29



def instance_norm(input_tensor,layer_name='instance_norm'):
    # takes a 4d tensor, batch size must be at least 1 ->  (*1*,256,256,3)
    with tf.variable_scope(layer_name):
        depth_of_tensor = input_tensor.get_shape()[3]
        scale = tf.get_variable("scale", [depth_of_tensor], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth_of_tensor], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        mean, variance = tf.nn.moments(input_tensor, axes=[1,2], keep_dims=True)
        stddevs = tf.sqrt(variance + 1e-6)
        instance_normalized = tf.divide((input_tensor - mean), stddevs)
        return scale*instance_normalized + offset

def conv2d(input_map, layer_name, kernel_size, stride, depth_out, padding):
    with tf.variable_scope(layer_name):

        input_shape = input_map.get_shape().as_list()

        weights = tf.get_variable("weights", [kernel_size, kernel_size, input_shape[3], depth_out], initializer=tf.random_normal_initializer(0.0, 0.02, dtype=tf.float32))
        conv_out = tf.nn.conv2d(input_map, weights, [1,stride,stride,1], padding)
        return conv_out



def residual_conv2d(input_map, layer_name, depth_out):
    with tf.variable_scope(layer_name):

        input_shape = input_map.get_shape().as_list()

        weights1 = tf.get_variable("weights1", [3, 3, input_shape[3], depth_out], initializer=tf.random_normal_initializer(0.0, 0.02, dtype=tf.float32))
        weights2 = tf.get_variable("weights2", [3, 3, input_shape[3], depth_out], initializer=tf.random_normal_initializer(0.0, 0.02, dtype=tf.float32))

        input_padded = tf.pad(input_map, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        conv_1_out = tf.nn.relu( instance_norm( tf.nn.conv2d(input_padded, weights1, [1,1,1,1], "VALID"), layer_name='instance_norm_1' ) )

        conv_1_out = tf.pad(conv_1_out, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        conv_2_out = instance_norm( tf.nn.conv2d(conv_1_out, weights2, [1,1,1,1], "VALID"), layer_name='instance_norm_2')

        return input_map + conv_2_out

def transposed_conv2d(input_map, layer_name, depth_out):
    with tf.variable_scope(layer_name):

        input_shape = input_map.get_shape().as_list()

        weights = tf.get_variable("weights", [3, 3, depth_out, input_shape[3]], initializer=tf.random_normal_initializer(0.0, 0.02, dtype=tf.float32))
        shape_of_next_layer = [input_shape[0], 2*input_shape[1], 2*input_shape[2], depth_out]
        conv_out = tf.nn.conv2d_transpose(input_map, weights, shape_of_next_layer, [1,2,2,1], "SAME")
        return conv_out
