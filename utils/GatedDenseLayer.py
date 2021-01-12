import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers import Dense


class GatedDenseLayer(Layer):
    def __init__(self, units, name=None, activation=None):
        super(GatedDenseLayer, self).__init__(name=name)
        self.units = units
        self.activation = activation
        self.h = Dense(units, trainable=True)
        self.g = Dense(units, trainable=True)

    def build(self, input_shape):
        self.h.build(input_shape)
        self.g.build(input_shape)
        super(GatedDenseLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        h = self.h(inputs)
        if self.activation is not None:
            h = self.activation(self.h(inputs))

        g = tf.nn.sigmoid(self.g(inputs))
        return tf.math.multiply(h, g)
