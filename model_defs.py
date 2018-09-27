"""In this file we define functions that will be used by the Model class.
More specifically, we define:
    1) The tensorflow graph parts that corresponds to the architecture of the network;
    2) A batch generator, specific for each type of NN architecture.
"""

import numpy as np
import tensorflow as tf


# FEEDFORWARD NETWORKS
# ----------- --------

def feedforward_nn(inputs):
    """Very simple feedforward network."""
    l1 = tf.layers.dense(inputs, 16)
    l2 = tf.layers.dense(l1    , 10)
    l3 = tf.layers.dense(l2    ,  6)
    return l3
            
def feedforward_batches_gen(xy_series, batch_size, window_size=None, shuffle=False):
    """Batch generator for a feedforward neural network.
    Generates batches of length `batch_size` from a list of (features, labels) tuples.
    """
    xy_indexes = np.arange(len(xy_series))
    if shuffle: np.random.shuffle(xy_indexes)
    for i in xy_indexes:
        x,y = xy_series[i]
        n_batches = (len(x) + batch_size - 1) // batch_size
        for b in range(n_batches): # ceil division to do not discard datapoints
            # slice features and labels to the current batch interval
            xx = x[batch_size*b : batch_size*(b+1)]
            yy = y[batch_size*b : batch_size*(b+1)]
            yield xx, yy


# CONVOLUTIONAL NETWORKS
# ------------- --------

def conv_nn(inputs, window_size, is_training=None, n_filters=64, kernel_size=3, strides=1, dense_size=128): # Bx1024x32 (supposing window_size=1024 and features=32)
    lconv = tf.layers.conv1d(inputs, filters=n_filters, kernel_size=kernel_size, strides=strides, activation=tf.nn.relu, padding='same') # Bx1024x64
    lpool = tf.layers.max_pooling1d(lconv, pool_size=2, strides=2, padding='same') # Bx512x64
    lflat = tf.reshape(lpool, [-1, (window_size/2)*n_filters/strides]) # Bx(512*64)
    ldense = tf.layers.dense(lflat, dense_size, activation=tf.nn.relu) # Bx128
    logits = tf.layers.dense(ldense, 6) # Bx6
    return logits

def conv_nn_2(inputs, window_size, is_training=None): # Bx1024x32 (supposing window_size=1024 and features=32)
    # b x ws x 32
    lconv = tf.layers.conv1d(inputs, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu, padding='same')
    # b x ws/2 x 32
    lconv2 = tf.layers.conv1d(lconv, filters=16, kernel_size=3, strides=2, activation=tf.nn.relu, padding='same')
    # b x ws/4 x 16
    lpool = tf.layers.max_pooling1d(lconv2, pool_size=2, strides=2, padding='same') # pooling does not affect channels
    # b x ws/8 x 16
    lflat = tf.reshape(lpool, [-1, (window_size/8)*16]) # flatten to apply dense layer
    ldense1 = tf.layers.dense(lflat, 256, activation=tf.nn.relu)
    ldense2 = tf.layers.dense(ldense1, 32, activation=tf.nn.relu) 
    logits = tf.layers.dense(ldense2, 6)
    return logits

def conv_nn_3(inputs, window_size, is_training): # Bx1024x32 (supposing window_size=1024 and features=32)
    """Same as 2 but with batch normalization.
    
    Arguments:
    - inputs: 
    - window_size:
    - is_training: boolean indicating if the network is training or not, used for batch normalization.
    """
    # b x ws x 32
    lconv = tf.layers.conv1d(inputs, filters=32, kernel_size=3, strides=2, activation=None, padding='same', use_bias=False)
    lconv = tf.layers.batch_normalization(lconv, training=is_training)
    lconv = tf.nn.relu(lconv)
    # b x ws/2 x 32
    lconv2 = tf.layers.conv1d(lconv, filters=16, kernel_size=3, strides=2, activation=None, padding='same', use_bias=False)
    lconv2 = tf.layers.batch_normalization(lconv2, training=is_training)
    lconv2 = tf.nn.relu(lconv2)
    # b x ws/4 x 16
    lpool = tf.layers.max_pooling1d(lconv2, pool_size=2, strides=2, padding='same') # pooling does not affect channels
    # b x ws/8 x 16
    lflat = tf.reshape(lpool, [-1, (window_size/8)*16]) # flatten to apply dense layer
    # b x ws*2
    ldense1 = tf.layers.dense(lflat, 256, activation=None, use_bias=False)
    ldense1 = tf.layers.batch_normalization(ldense1, training=is_training)
    ldense1 = tf.nn.relu(ldense1)
    # b x 256
    ldense2 = tf.layers.dense(ldense1, 32, activation=None, use_bias=False) 
    ldense2 = tf.layers.batch_normalization(ldense2, training=is_training)
    ldense2 = tf.nn.relu(ldense2)
    # b x 32
    logits = tf.layers.dense(ldense2, 6)
    # b x 6
    return logits

def strided_axis0(a, L):
    """Creates a view into the input array.
    Taken from https://stackoverflow.com/a/43413801/5103881
    """
    # Store the shape and strides info
    shp = a.shape
    s  = a.strides

    # Compute length of output array along the first axis
    nd0 = shp[0]-L+1

    # Setup shape and strides for use with np.lib.stride_tricks.as_strided
    # and get (n+1) dim output array
    shp_in = (nd0,L)+shp[1:]
    strd_in = (s[0],) + s
    return np.lib.stride_tricks.as_strided(a, shape=shp_in, strides=strd_in, writeable=False)

def conv_batches_gen(xy_series, batch_size, window_size, shuffle=False):
    """Generate batches of length `batch_size` from a list of (features, labels) tuples."""
    xy_indexes = np.arange(len(xy_series))
    if shuffle: np.random.shuffle(xy_indexes)
    for i in xy_indexes:
        x,y = xy_series[i]
        x_slices = strided_axis0(x, window_size)
        y_slices = y.iloc[window_size-1:] # we cannot fit any data point before (window_size - 1)
        n_batches = (len(x_slices) + batch_size - 1) // batch_size # trick to ceil using floor division
        for b in range(n_batches):
            yield x_slices[batch_size*b : batch_size*(b+1)], y_slices.iloc[batch_size*b : batch_size*(b+1)]