# # Simple script which includes functions for calculating surface loss in keras
# ## See the related discussion: https://github.com/LIVIAETS/boundary-loss/issues/14

from keras import backend as K
import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt as distance


def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)


def surface_loss_keras(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)


# # Scheduler
# ### The following scheduler was proposed by @marcinkaczor
# ### https://github.com/LIVIAETS/boundary-loss/issues/14#issuecomment-547048076

# class AlphaScheduler(Callback):
#     def __init__(self, alpha, update_fn):
#         self.alpha = alpha
#         self.update_fn = update_fn
#     def on_epoch_end(self, epoch, logs=None):
#         updated_alpha = self.update_fn(K.get_value(self.alpha))
#         K.set_value(self.alpha, updated_alpha)


# alpha = K.variable(1, dtype='float32')

# def gl_sl_wrapper(alpha):
#     def gl_sl(y_true, y_pred):
#         return alpha * generalized_dice_loss(
#             y_true, y_pred) + (1 - alpha) * surface_loss_keras(y_true, y_pred)
#     return gl_sl

# model.compile(loss=gl_sl_wrapper(alpha))

# def update_alpha(value):
#   return np.clip(value - 0.01, 0.01, 1)

# history = model.fit_generator(
#   ...,
#   callbacks=AlphaScheduler(alpha, update_alpha)
# )
