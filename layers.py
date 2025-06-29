import tensorflow as tf

def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):
    """Leaky ReLU activation function"""
    if alt_relu_impl:
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        # lrelu = 1/2 * (1 + leak) * x + 1/2 * (1 - leak) * |x|
        return f1 * x + f2 * tf.abs(x)
    else:
        return tf.maximum(x, leak * x)

def instance_norm(x, name="instance_norm"):
    """Instance normalization for TensorFlow 2.x"""
    epsilon = 1e-5
    mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    
    # Get the number of channels
    channels = x.shape[-1]
    
    # Create variables for scale and offset
    scale = tf.Variable(
        tf.random.truncated_normal([channels], mean=1.0, stddev=0.02),
        trainable=True,
        name=f"{name}_scale"
    )
    offset = tf.Variable(
        tf.zeros([channels]),
        trainable=True,
        name=f"{name}_offset"
    )
    
    normalized = (x - mean) / tf.sqrt(var + epsilon)
    return scale * normalized + offset

def general_conv2d(inputconv, o_d=32, f_h=3, f_w=3, s_h=1, s_w=1, stddev=0.02, 
                   padding="SAME", name="conv2d", do_norm=True, do_relu=True, relufactor=0):
    """General 2D convolution layer for TensorFlow 2.x"""
    
    # Create convolution layer
    conv = tf.nn.conv2d(
        inputconv,
        filters=tf.Variable(
            tf.random.truncated_normal([f_h, f_w, inputconv.shape[-1], o_d], stddev=stddev),
            name=f"{name}_weights"
        ),
        strides=[1, s_h, s_w, 1],
        padding=padding
    )
    
    # Add bias
    bias = tf.Variable(tf.zeros([o_d]), name=f"{name}_bias")
    conv = tf.nn.bias_add(conv, bias)
    
    # Apply normalization if requested
    if do_norm:
        conv = instance_norm(conv, name=f"{name}_norm")
    
    # Apply activation if requested
    if do_relu:
        if relufactor == 0:
            conv = tf.nn.relu(conv, name=f"{name}_relu")
        else:
            conv = lrelu(conv, relufactor, name=f"{name}_lrelu")
    
    return conv

def general_deconv2d(inputconv, outshape, o_d=32, f_h=3, f_w=3, s_h=1, s_w=1, 
                     stddev=0.02, padding="SAME", name="deconv2d", do_norm=True, 
                     do_relu=True, relufactor=0):
    """General 2D deconvolution (transpose convolution) layer for TensorFlow 2.x"""
    
    # Get input channels
    input_channels = inputconv.shape[-1]
    
    # Create transpose convolution
    conv = tf.nn.conv2d_transpose(
        inputconv,
        filters=tf.Variable(
            tf.random.truncated_normal([f_h, f_w, o_d, input_channels], stddev=stddev),
            name=f"{name}_weights"
        ),
        output_shape=outshape,
        strides=[1, s_h, s_w, 1],
        padding=padding
    )
    
    # Add bias
    bias = tf.Variable(tf.zeros([o_d]), name=f"{name}_bias")
    conv = tf.nn.bias_add(conv, bias)
    
    # Apply normalization if requested
    if do_norm:
        conv = instance_norm(conv, name=f"{name}_norm")
    
    # Apply activation if requested
    if do_relu:
        if relufactor == 0:
            conv = tf.nn.relu(conv, name=f"{name}_relu")
        else:
            conv = lrelu(conv, relufactor, name=f"{name}_lrelu")
    
    return conv