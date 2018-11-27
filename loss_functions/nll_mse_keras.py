import tensorflow as tf

def load():
    return tf_nll_mse

def tf_nll_mse(y_true, y_pred):
    y_var = y_pred[:,:1]
    y_mean = y_pred[:,1:]
    y_std  = tf.sqrt(tf.abs(y_var))

    return tf.keras.losses.mean_squared_error(y_true, y_mean)
