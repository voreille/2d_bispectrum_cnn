import numpy as np
import tensorflow as tf
from skimage.measure import label
from scipy.ndimage import watershed_ift


def post_processing(y_pred):
    if type(y_pred) != np.ndarray:
        y_pred_quantized = (y_pred.numpy() > 0.5).astype(np.uint8)
    else:
        y_pred_quantized = (y_pred > 0.5).astype(np.uint8)
    # y_pred_quantized = np.zeros_like(y_pred, dtype=np.uint8)
    # y_pred_quantized[..., 1] = (y_pred[..., 1] > 0.5).astype(np.uint8)
    # y_pred_quantized[..., 0] = (y_pred[..., 0] > 0.5).astype(np.uint8)
    # y_pred_quantized[..., 2] = (y_pred[..., 2] > 0.5).astype(np.uint8)
    batch_size = y_pred.shape[0]
    output = list()
    for s in range(batch_size):
        markers = label(y_pred_quantized[s, :, :, 0])
        markers[y_pred_quantized[s, :, :, 2] != 0] = -1
        out = watershed_ift((y_pred_quantized[s, :, :, 1]).astype(np.uint8),
                            markers)
        out[out == -1] = 0
        output.append(out)
    return np.stack(output, axis=0)