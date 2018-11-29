import numpy as np
import utils

def generate_sequence(length_std, length_mean, noise_std, length):
    count = 0
    state = 1
    sequence = np.array([])
    while count < length:
        l = int(np.random.normal(length_mean, length_std))
        n = np.random.normal(0, noise_std, l)
        s = np.ones(l)*state
        # add noise to signal
        y = n + s
        # change state
        state *= -1
        sequence = np.concatenate((sequence, y))
        count += l
    return sequence

def load_dataset(cfg):
    length_std  = float(cfg['length_std'])
    length_mean = float(cfg['length_mean'])
    noise_std   = float(cfg['noise_std'])
    length      = int(cfg['length'])
    nb_past_steps   = int(cfg['nb_past_steps'])
    nb_future_steps = int(cfg['nb_future_steps'])
    train_fraction  = float(cfg['train_fraction'])
    test_fraction   = float(cfg['test_fraction'])
    valid_fraction  = float(cfg['valid_fraction'])

    sequence = generate_sequence(length_std, length_mean, noise_std, length)

    xs, ys = utils.sequence_to_supervised(sequence, nb_past_steps, nb_future_steps)
    xs = np.expand_dims(xs, axis=2)
    ys = np.expand_dims(ys, axis=1)

    x_train, x_valid, x_test = utils.split_data(xs, train_fraction,
            valid_fraction, test_fraction)

    y_train, y_valid, y_test = utils.split_data(ys, train_fraction,
            valid_fraction, test_fraction)

    return x_train, y_train, x_valid, y_valid, x_test, y_test
