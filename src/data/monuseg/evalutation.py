import numpy as np
from scipy.ndimage import watershed_ift
from skimage.measure import label


def watershed_prediction(y_pred):
    markers = label(y_pred[..., 0])
    markers[y_pred[..., 2]] = -1
    y_pred = watershed_ift((y_pred[..., 1]).astype(np.uint8), markers)
    return y_pred
