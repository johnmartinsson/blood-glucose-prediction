import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import scipy
import seg

MOL_L_TO_MG_DL = 18.01801801801802

# numpy metrics
# TODO: rewrite this to simply load if from a csv file
def surveillance_error(targets, predictions):
    data = seg.load_seg_data("surveillance_error_grid_data/sheet4.xml")
    xs = np.linspace(0, 600, data.shape[0])
    ys = np.linspace(0, 600, data.shape[1])
    f = scipy.interpolate.interp2d(xs, ys, data)

    ma = np.max(data)
    scores = [f(t, p) for (t, p) in zip(targets, predictions)]
    return np.sum(scores)/(ma*len(targets))

def root_mean_squared_error(targets, predictions):
    return np.sqrt(np.mean(np.power(targets-predictions, 2)))
