import tensorflow as tf

def load(cfg):
    learning_rate = float(cfg['learning_rate'])
    return tf.keras.optimizers.Adam(learning_rate)
