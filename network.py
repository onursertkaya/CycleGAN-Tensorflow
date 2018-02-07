#!/usr/bin/python3

from tensor_ops import *

def generator(image, model_name, reuse):
    with tf.variable_scope(model_name, reuse=reuse):

        spatial_size_of_image = int( image.get_shape()[1] )

        image_padded = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]],"REFLECT")
        o1 = tf.nn.relu( instance_norm ( conv2d(image_padded, 'c7s1-32', 7, 1, 32, "VALID"), layer_name='gen_o1' ) )

        o2 = tf.nn.relu( instance_norm ( conv2d(o1, 'd64', 3, 2, 64, "SAME"), layer_name='gen_o2' ) )
        o3 = tf.nn.relu( instance_norm ( conv2d(o2, 'd128', 3, 2, 128, "SAME"), layer_name='gen_o3' ) )

        o4 = residual_conv2d(o3, 'R128_1', 128)
        o5 = residual_conv2d(o4, 'R128_2', 128)
        o6 = residual_conv2d(o5, 'R128_3', 128)
        o7 = residual_conv2d(o6, 'R128_4', 128)
        o8 = residual_conv2d(o7, 'R128_5', 128)
        o9 = residual_conv2d(o8, 'R128_6', 128)
        o10 = residual_conv2d(o9, 'R128_7', 128)
        o11 = residual_conv2d(o10, 'R128_8', 128)
        o12 = residual_conv2d(o11, 'R128_9', 128)

        o13 = tf.nn.relu( instance_norm ( transposed_conv2d(o12, 'u64', 64 ), layer_name='gen_o13' ) )
        o14 = tf.nn.relu( instance_norm ( transposed_conv2d(o13, 'u32', 32 ), layer_name='gen_o14' ) )

        o14_padded = tf.pad(o14, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        o15 = conv2d(o14_padded, 'c7s1-3', 7, 1, 3, "VALID")

        final_bias = tf.get_variable('final_bias', [3], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        o16 = o15 + final_bias
        return tf.nn.tanh(o16)

# 70x70 patchGAN ~ effectively...
# after 3 x stride2 + 1 x stride1
# effective receptive field is 70x70
# calculations done at https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m
def discriminator(image, model_name, reuse):
    with tf.variable_scope(model_name, reuse=reuse):
        # get image shapes
        input_image_shape = image.get_shape().as_list()

        first_bias = tf.get_variable('first_bias', [input_image_shape[0], input_image_shape[1]/2,input_image_shape[1]/2, 64], initializer=tf.constant_initializer(0.0), dtype=tf.float32 )
        o1 = tf.nn.leaky_relu( conv2d(image,'C64', 4, 2, 64, "SAME") + first_bias )
        # o1 (128x128x64)
        o2 = tf.nn.leaky_relu( instance_norm ( conv2d(o1, 'C128', 4, 2, 128, "SAME"), layer_name='disc_o2') )
        # o2 (64x64x128)
        o3 = tf.nn.leaky_relu( instance_norm ( conv2d(o2, 'C256', 4, 2, 256, "SAME"), layer_name='disc_o3') )
        # o3 (32x32x256)
        o4 = tf.nn.leaky_relu( instance_norm ( conv2d(o3, 'C512', 4, 1, 512, "SAME"), layer_name='disc_o4') )
        # o4 (32x32x512)
        o5 = conv2d(o4, 'C1', 4, 1, 1, "SAME")
        # o5 (32x32x1)

        final_bias = tf.get_variable('final_bias', [1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        o6 = o5 + final_bias


        return o6
