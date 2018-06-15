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

def sequence_to_supervised_all(data, nb_past_steps, nb_future_steps):
    """Computes feature and target data for the sequence. The features are the
    number of past steps used, and the targets are all the future values until
    the number of steps into the future specified."""
    x = rolling_window(data, nb_past_steps)
    y = rolling_window(data, nb_future_steps)

    x, y = zip(*zip(x[:-nb_future_steps], y[nb_past_steps:]))

    return np.array(x), np.array(y)

def walk_forward_array(data, nb_past_steps, nb_future_steps, step_size,
        nb_steps, nb_training_samples, nb_validation_samples, nb_test_samples):
    training_array = np.zeros((nb_steps, nb_training_samples))
    validation_array = np.zeros((nb_steps, nb_validation_samples))
    test_array = np.zeros((nb_steps, nb_test_samples))

    for i_step, (train, valid, test) in enumerate(walk_forward(data, step_size,
        nb_steps, nb_training_samples, nb_validation_samples, nb_test_samples)):
        x_train, y_train = sequence_to_supervised_all(train, nb_past_steps,
                nb_future_steps)
        x_valid, y_valid = sequence_to_supervised_all(valid, nb_past_steps,
                nb_future_steps)
        x_test, y_test = sequence_to_supervised_all(test, nb_past_steps,
                nb_future_steps)
        training_array[i_step] = train
        validation_array[i_step] = valid
        test_array[i_step] = test

    return training_array, validation_array, test_array

def walk_forward(data, step_size, nb_steps, nb_training_samples,
        nb_validation_samples, nb_test_samples):
    """Walk-forward procedure to generate training, validation and test data
    from a sequence."""
    if step_size*(nb_steps-1) + nb_training_samples + nb_validation_samples + nb_test_samples > len(data):
        raise UserWarning("can not fit all steps into the sequence ...")

    start_idx          = 0
    training_end_idx   = start_idx + nb_training_samples
    validation_end_idx = training_end_idx + nb_validation_samples
    test_end_idx       = validation_end_idx + nb_test_samples

    i_step = 0
    while test_end_idx < len(data) and i_step < nb_steps:
        training_sequence   = data[start_idx:training_end_idx]
        validation_sequence = data[training_end_idx:validation_end_idx]
        test_sequence       = data[validation_end_idx:test_end_idx]

        start_idx          += step_size
        training_end_idx   += step_size
        validation_end_idx += step_size
        test_end_idx       += step_size

        i_step += 1
        yield training_sequence, validation_sequence, test_sequence
