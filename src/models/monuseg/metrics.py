from skimage.segmentation import watershed
import numpy as np


def combine_seg_contour(y_pred):
    return watershed(y_pred[..., 1], mask=y_pred[..., 0])[..., np.newaxis]

def f1_score(y_true, y_pred):
    colors = np.unique(y_true.reshape(-1, y_true.shape[2]), axis=0)
    pass
