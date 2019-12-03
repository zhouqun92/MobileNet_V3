# -*- coding: utf-8 -*-
"""Implementation of Mobilenet V3 by tf.slim.
Architecture: https://arxiv.org/pdf/1905.02244.pdf
"""
import tensorflow as tf 
import tensorflow.contrib.slim as slim
def relu6(x, name='relu6'):
    return tf.nn.relu6(x, name)
def hard_swish(x, name='hard_swish'):
    with tf.variable_scope(name):
        h_swish = x * tf.nn.relu6(x + 3) / 6
    return h_swish
def hard_sigmoid(x, name='hard_sigmoid'):
    with tf.variable_scope(name):
        h_sigmoid = tf.nn.relu6(x + 3) / 6
    return h_sigmoid
def _squeeze_excitation_layer(input, out_dim, ratio,is_training=True, reuse=None):
    squeeze = slim.avg_pool2d(input,input.get_shape()[1:-1], stride=1)
    excitation=slim.convolution2d(squeeze,int(out_dim / ratio),[1,1],stride=1,activation_fn=relu6)
    excitation=slim.convolution2d(excitation,out_dim,[1,1],stride=1,activation_fn=hard_sigmoid)
    excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
    scale = input * excitation
    return scale
def mobilenet_v3_block(input, kernel, batch_norm_params,expansion_dim, output_dim, stride, name, is_training=True,
                       shortcut=True, activatation="RE", ratio=16, se=False):
    if activatation == "HS":
        activation_fn= hard_swish
    elif activatation == "RE":
        activation_fn= relu6
    with tf.variable_scope(name):
        with slim.arg_scope([slim.convolution2d, slim.separable_conv2d], \
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            biases_initializer=tf.zeros_initializer(),
                            #weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            padding='SAME'):
            net=slim.convolution2d(input,expansion_dim,[1,1],stride=1,activation_fn=activation_fn)
            net=slim.separable_convolution2d(net,num_outputs=None, kernel_size=kernel,depth_multiplier=1,stride=stride,activation_fn=activation_fn)    
            if se:
                channel = net.get_shape().as_list()[-1]
                net = _squeeze_excitation_layer(net, out_dim=channel, ratio=ratio)                                      
            net=slim.convolution2d(net,output_dim,[1,1],stride=1,activation_fn=None)
            if shortcut and stride == 1:
                net += input
            return net
def mobilenet_v3(inputs,classes_num,multiplier=1.0, is_training=True,type='small'):
    end_points = {}
    if type=='small':
        layers = [
            [16, 16, 3, 2, "RE", True, 16],
            [16, 24, 3, 2, "RE", False, 72],
            [24, 24, 3, 1, "RE", False, 88],
            [24, 40, 5, 2, "RE", True, 96],
            [40, 40, 5, 1, "RE", True, 240],
            [40, 40, 5, 1, "RE", True, 240],
            [40, 48, 5, 1, "HS", True, 120],
            [48, 48, 5, 1, "HS", True, 144],
            [48, 96, 5, 2, "HS", True, 288],
            [96, 96, 5, 1, "HS", True, 576],
            [96, 96, 5, 1, "HS", True, 576],
        ]
    else:
        layers = [
            [16, 16, 3, 1, "RE", False, 16],
            [16, 24, 3, 2, "RE", False, 64],
            [24, 24, 3, 1, "RE", False, 72],
            [24, 40, 5, 2, "RE", True, 72],
            [40, 40, 5, 1, "RE", True, 120],

            [40, 40, 5, 1, "RE", True, 120],
            [40, 80, 3, 2, "HS", False, 240],
            [80, 80, 3, 1, "HS", False, 200],
            [80, 80, 3, 1, "HS", False, 184],
            [80, 80, 3, 1, "HS", False, 184],

            [80, 112, 3, 1, "HS", True, 480],
            [112, 112, 3, 1, "HS", True, 672],
            [112, 160, 5, 1, "HS", True, 672],
            [160, 160, 5, 2, "HS", True, 672],
            [160, 160, 5, 1, "HS", True, 960],
        ]
    batch_norm_params = {
        'decay': 0.999,
        'epsilon': 0.001,
        'updates_collections':  None,#tf.GraphKeys.UPDATE_OPS,
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
        'is_training': is_training
    }
    input_size = inputs.get_shape().as_list()[1:-1]
    assert ((input_size[0] % 32 == 0) and (input_size[1] % 32 == 0))
    reduction_ratio = 4
    x=slim.convolution2d(inputs,int(16*multiplier),[3,3],stride=2,activation_fn=hard_swish,normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,biases_initializer=None)           
    with tf.variable_scope("MobilenetV3"):
        for idx, (in_channels, out_channels, kernel_size, stride, activatation, se, exp_size) in enumerate(layers):
            in_channels = int(in_channels * multiplier)
            out_channels = int(out_channels * multiplier)
            exp_size = int(exp_size * multiplier)
            x = mobilenet_v3_block(x, [kernel_size,kernel_size],batch_norm_params,exp_size, out_channels, stride,
                                   "bneck{}".format(idx), is_training=is_training,
                                   shortcut=(in_channels==out_channels), activatation=activatation,
                                   ratio=reduction_ratio, se=se)
            end_points["bneck{}".format(idx)] = x
        if type=='small':
            conv1_out = int(576 * multiplier)
        else:
            conv1_out = int(960 * multiplier)
        x=slim.convolution2d(x,conv1_out,[1,1],stride=1,normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,activation_fn=hard_swish)
        if type=='small':
            x = _squeeze_excitation_layer(x, out_dim=conv1_out, ratio=reduction_ratio, 
                                     is_training=is_training, reuse=None)
        end_points["conv1_out_1x1"] = x
        x = slim.avg_pool2d(x,x.get_shape()[1:-1], stride=1)
        #x = hard_swish(x)
        end_points["global_pool"] = x
    with tf.variable_scope('Logits_out'):
        conv2_out = int(1280 * multiplier)
        x=slim.convolution2d(x,conv2_out,[1,1],stride=1,activation_fn=hard_swish)
        end_points["conv2_out_1x1"] = x
        x=slim.convolution2d(x,classes_num,[1,1],stride=1,activation_fn=None)
        logits = tf.layers.flatten(x)
        logits = tf.identity(logits, name='output')
        end_points["Logits_out"] = logits
    return logits, end_points



if __name__ == "__main__":
    input_test = tf.ones([1, 224, 224, 3])
    num_classes = 1000
    model, end_points = mobilenet_v3(input_test, num_classes, multiplier=1.0, is_training=True,type='large')
