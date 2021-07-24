from itertools import product

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

from src.models.layers import ECHConv2D

np.set_printoptions(precision=2, linewidth=150)

x = tf.random.uniform((32, 128, 128, 4))
layer = ECHConv2D(1, 9, initializer=tf.keras.initializers.Constant(1.0))

y = layer(x)
