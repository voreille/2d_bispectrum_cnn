import numpy as np


def f_score(y_true, y_pred):
    colors = np.unique(y_true.reshape(-1, y_true.shape[2]), axis=0)
    for c_index in range(colors.shape[0]):
        color = colors[c_index, :]
        nuclei_mask = np.all(y_true == color, axis=-1)
        intersection = y_pred * nuclei_mask
        