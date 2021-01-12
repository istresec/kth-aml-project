import tensorflow as tf


def reparameterize(inputs):
    mean, logvar = inputs
    eps = tf.keras.backend.random_normal(shape=tf.shape(mean))
    return mean + tf.exp(0.5 * logvar) * eps
