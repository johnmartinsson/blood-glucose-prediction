import tensorflow as tf

def load():
    return tf_mse

def tf_mse(targets, predictions):
    return tf.keras.losses.mean_squared_error(targets, predictions)
