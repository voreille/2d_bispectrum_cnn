from itertools import product

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

from src.models.layers import CHConv2D


def get_sh(kernel_size=3, degrees=5):
    radius = (kernel_size - 1) // 2
    x_grid = np.arange(-radius, radius + 1, 1)
    x, y = np.meshgrid(x_grid, x_grid)
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    window = np.zeros_like(r)
    window[r**2 < radius**2] = 1
    atoms = np.zeros(
        (
            degrees + 1,
            kernel_size,
            kernel_size,
        ),
        dtype=np.csingle,
    )
    for n in range(degrees + 1):
        atoms[n, :, :] = np.exp(theta * n * 1j)
    return atoms * window


degrees = 5
kernel_size=9
shs = get_sh(kernel_size=kernel_size, degrees=degrees)




factor = np.zeros((degrees + 1, ))
factor[1] = 1
factor[3] = 1.5
factor[5] = 2
factor = factor[:, np.newaxis, np.newaxis]


f = np.real(np.sum(shs * factor, axis=0))




layer = CHConv2D(1, degrees, kernel_size, initializer=tf.keras.initializers.Constant(1.0))


x = f[np.newaxis, :, :, np.newaxis]
y = layer(x).numpy()


np.squeeze(y)
print(y)





