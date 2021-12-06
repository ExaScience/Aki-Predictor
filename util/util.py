import sys

sys.path.append('../')
import os
import threading

import keras
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd

'''
Config GPU
'''


def config_gpu(using_config=True, gpu = '1'):
    if using_config:
        print('Using config GPU!')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        # Config minimize GPU with model
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)
    else:
        print('Not using config GPU!')


'''
Create folder if not exist
'''


def create_folder(path_folder):
    if not os.path.exists(path_folder):
        os.makedirs((path_folder))
        print('Directory {} created successfully!'.format(path_folder))
    else:
        print('Directory {} already exists!'.format(path_folder))


'''
Write data to csv file
'''


def write_csv(data, path_file):
    if not os.path.exists(path_file):
        with open(path_file, 'w') as f:
            data.to_csv(f, encoding='utf-8', header=True, index=False)

