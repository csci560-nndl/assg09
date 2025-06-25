import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
import numpy as np
import sklearn as sk
import matplotlib as mp
import scipy as sp
import pandas as pd
import tensorflow as tf

print('Python version    : ', sys.version)
print('numpy version     : ', np.__version__)
print('scikit-learn  vers: ', sk.__version__)
print('matplotlib version: ', mp.__version__)
print('scipy version     : ', sp.__version__)
print('pandas version    : ', pd.__version__)
print('tensorflow version: ', tf.__version__)

dev = tf.config.list_physical_devices()
print('Physical Devices  : ', dev)
dev = tf.config.list_logical_devices()
print('Available Devices : ', dev)