import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import scipy

# numpy metrics
def surveillance_error(targets, predictions):
    data = np.loadtxt('seg.csv')

    xs = np.linspace(0, 600, data.shape[0])
    ys = np.linspace(0, 600, data.shape[1])
    f = scipy.interpolate.interp2d(xs, ys, np.transpose(data))

    ma = np.max(data)
    scores = [f(t, p) for (t, p) in zip(targets, predictions)]
    # print(scores)
    return np.sum(scores)/(ma*len(targets))

def root_mean_squared_error(targets, predictions):
    return np.sqrt(np.mean(np.power(targets-predictions, 2)))
