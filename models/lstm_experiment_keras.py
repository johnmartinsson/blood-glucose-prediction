import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

def get_activation_function_by_name(name='exp'):
    if name == 'exp':
        return tf.exp
    elif name == 'softplus':
        return tf.math.softplus
    elif name == 'relu':
        return tf.nn.relu
    else:
        raise ValueError("Actiation function: {} is not supported.".format(name))

def load(input_shape, output_shape, cfg):
    nb_lstm_states = int(cfg['nb_lstm_states'])


    inputs = KL.Input(shape=input_shape)
    x = KL.CuDNNLSTM(units=nb_lstm_states, unit_forget_bias=True)(inputs)

    x = KL.Dense(512)(x)
    x = KL.Activation('relu')(x)
    x = KL.Dropout(0.2)(x)

    x = KL.Dense(256)(x)
    x = KL.Activation('relu')(x)
    x = KL.Dropout(0.3)(x)

    mu = KL.Dense(1)(x)
    std = KL.Dense(1)(x)
    activation_fn = get_activation_function_by_name(cfg['activation_function'])
    std = KL.Activation(activation_fn, name="exponential_activation")(std)

    output = KL.Concatenate(axis=-1)([std, mu])
    model = KM.Model(inputs=[inputs], outputs=[output])

    return model
