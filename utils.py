import numpy as np

def dataframe_to_supervised(df, nb_past_steps, nb_steps_in_future):
    xs = []
    ys = []
    for column in df.columns:
        sequence = df[column].as_matrix()
        x, y = sequence_to_supervised(sequence, nb_past_steps,
                nb_steps_in_future)
        x = x.reshape(x.shape+(1,))
        y = y.reshape(y.shape+(1,))
        xs.append(x)
        ys.append(y)

    return np.concatenate(xs, axis=2), np.concatenate(ys, axis=1)[:,0]

def rolling_window(sequence, window):
    """Splits the sequence into window sized chunks by moving the window along
    the sequence."""
    shape = sequence.shape[:-1] + (sequence.shape[-1] - window + 1, window)
    strides = sequence.strides + (sequence.strides[-1],)
    return np.lib.stride_tricks.as_strided(sequence, shape=shape, strides=strides)

def sequence_to_supervised(data, nb_past_steps, nb_steps_in_future):
    """Computes feature and target data for the sequence. The features are the
    number of past steps used, and the target is the specified number of steps
    into the future."""
    x = rolling_window(data, nb_past_steps)
    y = data[nb_past_steps+nb_steps_in_future-1::1]

    x, y = zip(*zip(x, y))

    return np.array(x), np.array(y)

def split_data(xs, train_fraction, valid_fraction, test_fraction):
    n = len(xs)
    nb_train = int(np.ceil(train_fraction*n))
    nb_valid = int(np.ceil(valid_fraction*n))
    i_end_train = nb_train
    i_end_valid = nb_train+nb_valid

    return np.split(xs, [i_end_train, i_end_valid])



def sequence_to_supervised_all(data, nb_past_steps, nb_future_steps):
    """Computes feature and target data for the sequence. The features are the
    number of past steps used, and the targets are all the future values until
    the number of steps into the future specified."""
    x = rolling_window(data, nb_past_steps)
    y = rolling_window(data, nb_future_steps)

    x, y = zip(*zip(x[:-nb_future_steps], y[nb_past_steps:]))

    return np.array(x), np.array(y)
