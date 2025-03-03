"""
losses.py - custom loss functions for the AutoEncoder.
"""

import tensorflow as tf

def mssim_loss(dynamic_range):
    """
    Multi-Scale SSIM loss for reconstruction.
    Returns 1 - MSSIM as the loss, so higher MSSIM => lower loss.
    """
    def loss(y_true, y_pred):
        # tf.image.ssim_multiscale returns a value between 0 and 1
        # We want to minimize 1 - MSSIM
        return 1.0 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, dynamic_range))
    return loss

# If you ever need them, you can re-add or uncomment:
# def ssim_loss(dynamic_range):
#     def loss(y_true, y_pred):
#         return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, dynamic_range))
#     return loss

# def l2_loss(y_true, y_pred):
#     return tf.reduce_mean(tf.square(y_true - y_pred))