import tensorflow as tf

def load():
    return tf_negative_log_normal_density_loss

def tf_negative_log_normal_density_loss(y_true, y_pred):
    n = y_pred.shape[1]
    y_mean = y_pred[:,n//2:]
    y_var  = y_pred[:,:n//2]
    y_std  = tf.sqrt(tf.abs(y_var)) # TODO: This is a hack!

    dist = tf.distributions.Normal(loc=y_mean, scale=y_std)
    probs = dist.prob(y_true)
    return tf.reduce_mean(-tf.log(tf.keras.backend.epsilon() + probs)) # TODO: This is a hack!


