import tensorflow as tf

def load(learning_rate):
    return tf.keras.optimizers.Adam(learning_rate)
