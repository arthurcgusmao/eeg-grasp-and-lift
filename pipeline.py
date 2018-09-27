"""This file should be called to run the model. Code present here is the same as the one shown in the jupyter notebook. It was moved here so that we are able to run it from the command line.
"""

import os
import numpy as np

np.random.seed(30) # fix random seed
os.environ["CUDA_VISIBLE_DEVICES"] = '1' # fix cuda device to be used

## Read data, standardize, and separate into train/valid

from helpers import read_and_preprocess_data

train, valid = read_and_preprocess_data()

## Instantiate model and fit

from model import Model
from model_defs import conv_nn_2, conv_nn_3, conv_batches_gen

nn = Model(conv_nn_2, window_size=1024)
nn.set_data(train, valid)
nn.fit(
    epochs=8,
    batch_size=1024,
    batches_gen=conv_batches_gen,
)