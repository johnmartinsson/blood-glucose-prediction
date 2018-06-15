import tensorflow as tf
import os

def train(model, x_train, y_train, x_valid, y_valid, batch_size, epochs,
        patience, shuffle, artifacts_path):
    history = model.fit(
        x_train,
        y_train,
        validation_data = (x_valid, y_valid),
        epochs          = epochs,
        batch_size      = batch_size,
        shuffle         = shuffle,
        callbacks       = [
            tf.keras.callbacks.EarlyStopping(
                monitor  = 'val_loss',
                patience = patience,
                mode     = 'min'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=artifacts_path
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath       = os.path.join(artifacts_path, "model.hdf5"),
                monitor        = 'val_loss',
                save_best_only = True,
                period         = 30
            )
        ]
    )
    return model
