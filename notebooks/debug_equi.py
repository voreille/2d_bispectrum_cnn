import os
import random
from itertools import product

import tensorflow as tf
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

from src.models.layers import ECHConv2D

np.set_printoptions(precision=2, linewidth=150)
os.environ['PYTHONHASHSEED'] = '0'
random.seed(12345)
seed(1)
tf.random.set_seed(2)

x = np.random.uniform(size=(9, 9))
x_rotated = rotate(x, 90, reshape=False)
x = np.stack([x, x_rotated], axis=0)[..., np.newaxis]
layer = ECHConv2D(1, 9, initializer=tf.keras.initializers.Constant(1.0))

y = layer(x)

print(y)
