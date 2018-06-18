import tensorflow as tf

def tf_gmse(y_true, y_pred):
    return tf.reduce_mean(tf_penalty(y_true, y_pred)*tf.square(y_true-y_pred))

def load():
    return tf_gmse

def tf_penalty(y, y_hat):
    aL = tf.constant(1.5)
    aH = tf.constant(1.0)
    BL = tf.constant(90.0)
    BH = tf.constant(300.0)
    YL = tf.constant(30.0)
    YH = tf.constant(60.0)
    TL = tf.constant(85.0)
    TH = tf.constant(155.0)
    t1 = aL*tf_sigmoid_(y, TL, BL)*tf_sigmoid(y_hat, y, YL)
    t2 = aH*tf_sigmoid(y, TH, BH)*tf_sigmoid_(y_hat, y, YH)

    return 1 + t1 + t2

# Reference:
# http://www.clear-lines.com/blog/post/S-shaped-market-adoption-curve.aspx
def tf_sigmoid(x, a, e):
    f1 = tf.constant(0.2)
    f2 = tf.constant(0.8)
    t1 = a
    t2 = a+e

    alpha = (tf.log(1.0/f1-1)-tf.log(1.0/f2-1))/(t2-t1)
    x0 = tf.log(1.0/f1 - 1)/alpha + t1

    return 1/(1+tf.exp(-(x-x0)*alpha))

def tf_sigmoid_(x, a, e):
    f1 = tf.constant(0.8)
    f2 = tf.constant(0.2)
    t1 = a-e
    t2 = a

    alpha = (tf.log(1.0/f1-1)-tf.log(1.0/f2-1))/(t2-t1)
    x0 = tf.log(1.0/f1 - 1)/alpha + t1

    return 1/(1+tf.exp(-(x-x0)*alpha))
