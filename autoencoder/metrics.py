"""
metrics.py - custom metrics for evaluating the reconstruction quality of the AutoEncoder.
Updated for Python 3.12 and modern TensorFlow.
"""

import tensorflow as tf
from keras import backend as K

def mssim_metric(dynamic_range):
    """
    Multi-Scale SSIM metric to evaluate image similarity.

    Args:
        dynamic_range (float): Dynamic range of the pixel values.

    Returns:
        A function that computes the mean multi-scale SSIM over image channels.
    """
    def mssim(imgs_true, imgs_pred):
        # Compute multi-scale SSIM and average over channels
        return K.mean(tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range), axis=-1)

    # Rename so Keras logs this metric under the key "mssim" (and "val_mssim")
    mssim.__name__ = "mssim"
    return mssim