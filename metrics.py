import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
# import scipy
# import seg

MOL_L_TO_MG_DL = 18.01801801801802

# numpy metrics
# TODO: rewrite this to simply load if from a csv file
# def surveillance_error(targets, predictions):
    # data = seg.load_seg_data("surveillance_error_grid_data/sheet4.xml")
    # xs = np.linspace(0, 600, data.shape[0])
    # ys = np.linspace(0, 600, data.shape[1])
    # f = scipy.interpolate.interp2d(xs, ys, data)

    # ma = np.max(data)
    # scores = [f(t, p) for (t, p) in zip(targets, predictions)]
    # return np.sum(scores)/(ma*len(targets))

def root_mean_squared_error(targets, predictions):
    return np.sqrt(np.mean(np.power(targets-predictions, 2)))

# tensorflow loss functions
def tf_gmse(y_true, y_pred):
    return tf.reduce_mean(tf_penalty(y_true, y_pred)*tf.square(y_true-y_pred))

def tf_negative_log_gaussian_probability_loss(y_true, y_pred):
    n = y_pred.shape[1]
    y_mean = y_pred[:,:n//2]
    y_var  = y_pred[:,n//2:]
    y_std  = tf.sqrt(tf.abs(y_var)) # TODO: This is a hack!

    dist = tf.distributions.Normal(loc=y_mean, scale=y_std)
    probs = dist.prob(y_true)
    return tf.reduce_mean(-tf.log(tf.keras.backend.epsilon() + probs)) # TODO: This is a hack!

def tf_penalty(y, y_hat):
    aL = tf.constant(1.5)
    aH = tf.constant(1.0)
    BL = tf.constant(30.0)
    BH = tf.constant(100.0)
    YL = tf.constant(10.0)
    YH = tf.constant(20.0)
    TL = tf.constant(85.0)
    TH = tf.constant(155.0)
    t1 = aL*tf_sigmoid_(y, TL, BL)*tf_sigmoid(y_hat, y, YL)
    t2 = aH*tf_sigmoid(y, TH, BH)*tf_sigmoid_(y_hat, y, YH)

    return 1 + t1 + t2

# Reference:
# http://www.clear-lines.com/blog/post/S-shaped-market-adoption-curve.aspx
def tf_sigmoid(x, a, e):
    f1 = tf.constant(0.001)
    f2 = tf.constant(0.999)
    t1 = a
    t2 = a+e

    alpha = (tf.log(1.0/f1-1)-tf.log(1.0/f2-1))/(t2-t1)
    x0 = tf.log(1.0/f1 - 1)/alpha + t1

    return 1/(1+tf.exp(-(x-x0)*alpha))

def tf_sigmoid_(x, a, e):
    f1 = tf.constant(0.999)
    f2 = tf.constant(0.001)
    t1 = a-e
    t2 = a

    alpha = (tf.log(1.0/f1-1)-tf.log(1.0/f2-1))/(t2-t1)
    x0 = tf.log(1.0/f1 - 1)/alpha + t1

    return 1/(1+tf.exp(-(x-x0)*alpha))
