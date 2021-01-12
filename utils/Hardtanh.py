import tensorflow as tf


class Hardtanh():
    """
    Hardtanh function implementation, used as an activation function
    """

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.slope = tf.constant((max_val - min_val) / 2., 'float32')
        self.offset = tf.constant((min_val + max_val) / 2., 'float32')

    def __call__(self, x):
        x = tf.multiply(x, self.slope)
        x = tf.add(x, self.offset)
        x = tf.clip_by_value(x, self.min_val, self.max_val)
        return x
