# Basic Code is taken from https://github.com/ckmarkoh/GAN-tensorflow

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

import os
import shutil
from PIL import Image
import time
import random
import torch.nn as nn

from layers import *

img_height = 28
img_width = 28
img_layer = 1
img_size = img_height * img_width


batch_size = 1
pool_size = 50
ngf = 32
ndf = 64





def build_resnet_block(inputres, dim, name="resnet"):
    
    with tf.compat.v1.variable_scope(name):

        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID","c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID","c2",do_relu=False)
        
        return tf.nn.relu(out_res + inputres)

    

def build_generator_MNIST_without_resnet(inputgen, name="generator"):
    with tf.compat.v1.variable_scope(name):
        # First conv: input 1 channel -> 64 channels
        o_c1 = general_conv2d(inputgen, 64, 4, 4, 2, 2, 0.02, "SAME", "cv1")
        
        # Second conv: 64 ->128 channels
        o_c2 = general_conv2d(o_c1, 128, 4, 4, 2, 2, 0.02, "SAME", "cv2")

        # Third conv: 128 ->128 channels 
        o_c3 = general_conv2d(o_c2, 128, 3, 3, 1, 1, 0.02, "SAME", "conv3")

        # Fourth conv: 128 -> 128 channels
        o_c4 = general_conv2d(o_c3, 128, 3, 3, 1, 1, 0.02, "SAME", "conv4")

        
        # first deconv: 7x7 -> 14x14
        o_d1 = general_deconv2d(o_c4, [batch_size, 14, 14, 64],
                        64, 4, 4, 2, 2, 0.02, "SAME", "decv1")

        # second deconv: 14x14 -> 28x28
        o_d2 = general_deconv2d(o_d1, [batch_size, 28, 28, 1],
                        1, 4, 4, 2, 2, 0.02, "SAME", "decv2", do_relu=False)

        # Output activation
        out_gen = tf.nn.tanh(o_d2, "tanh_out")

        return out_gen



def build_generator_MNIST(inputgen, name="generator"):
    with tf.compat.v1.variable_scope(name):
        # inputgen: [batch, 28, 28, 1]

        # 1) Erste Conv: 1 -> 64 channels, stride=1 (gets 28x28)
        o_c1 = general_conv2d(inputgen, 64, 7, 7, 1, 1, 0.02, "SAME", "cv1")
        
        # 2) Downsampling: 64 -> 128, stride=2 (28x28 -> 14x14)
        o_c2 = general_conv2d(o_c1, 128, 3, 3, 2, 2, 0.02, "SAME", "cv2")
        
        # 3) ResNet-Blocks on 14x14-Features
        
        r_1 = build_resnet_block(o_c2, 128, name=f"resnet_{1}")

        r_2 = build_resnet_block(r_1, 128, name=f"resnet_{2}")

        r_3 = build_resnet_block(r_2, 128, name=f"resnet_{3}")
        

        # 4) Upsampling: 128 -> 64, 14x14 -> 28x28
        batch_size = tf.shape(inputgen)[0]
        o_d1 = general_deconv2d(r_3, [batch_size, 28, 28, 64],
                        64, 3, 3, 2, 2, 0.02, "SAME", "decv1")
        
        # 5) Final Conv on 1 channel (28x28 -> 28x28)
        o_c3 = general_conv2d(o_d1, 1, 7, 7, 1, 1, 0.02, "SAME", "cv3", do_relu=False)
        
        # 6) Tanh-Ausgabe
        out_gen = tf.nn.tanh(o_c3, "tanh_out")
        
        return out_gen


def build_discriminator_MNIST(inputdisc, name="discriminator"):
    with tf.compat.v1.variable_scope(name):
        # First conv: 1 -> 64 channels
        o_c1 = general_conv2d(inputdisc, 64, 4, 4, 2, 2, 0.02, "SAME", "cv1", do_norm=False, relufactor=0.2)
       
        # Second conv: 64 ->128 channels
        o_c2 = general_conv2d(o_c1, 128, 4, 4, 2, 2, 0.02, "SAME", "cv2", relufactor=0.2)
       
        # Third conv: 128 -> 256 channels
        o_c3 = general_conv2d(o_c2, 256, 4, 4, 2, 2, 0.02, "SAME", "conv3", relufactor=0.2)
        
        # Final conv: 256 ->1 channel
        o_fc = general_conv2d(o_c3, 1, 4, 4, 1, 1, 0.02, "VALID", "fc", do_norm=False, do_relu=False)
        
        return o_fc

