import numpy as np
import tensorflow as tf
import metrics

def load(input_shape, output_shape, cfg):
    nb_lstm_states = int(cfg['nb_lstm_states'])
    # design network
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(
        units=nb_lstm_states,
        input_shape=input_shape,
        unit_forget_bias=True,
    ))
    model.add(tf.keras.layers.Dense(
        units=output_shape
    ))
    return model
