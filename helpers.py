import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Data Preprocessing

## Read data

def read_data(sbj_i, srs_i):
    x = pd.read_csv('./data/train/subj{}_series{}_data.csv'  .format(sbj_i, srs_i), index_col='id')
    y = pd.read_csv('./data/train/subj{}_series{}_events.csv'.format(sbj_i, srs_i), index_col='id')
    return x,y

def read_all_data():
    print('Reading data...'),
    data = [read_data(i,j) for j in range(1,9) for i in range(1,13)]
    print('Data read. There are {} series in total.'.format(len(data)))
    return data


## Standardization

def standardize_data(data):
    print('Standardizing data...'),
    scaler = StandardScaler()

    all_x = pd.DataFrame()
    for x,_ in data:
        all_x = pd.concat((x, all_x))

    scaler.fit(all_x)
    data_std = []
    for x,y in data:
        data_std.append([scaler.transform(x),y])
        
    print('Data standardized.')
    return data_std


## Separate Training and Validation data

def split_train_valid(data):
    print('Separating data into train/valid...'),
    train = data
    # We randomly select one series from each subject to be part of the validation set.
    valid_idxs = reversed([0,14,23,25,37,44,50,57,65,78,80,94])
    valid = [train.pop(i) for i in valid_idxs]
    print('Data separated into train/valid.')
    return train, valid


## Perform all preprocessing at once

def read_and_preprocess_data():
    """Read, standardize and split data.
    """
    data = read_all_data()
    data = standardize_data(data)
    return split_train_valid(data)