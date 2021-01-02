import tensorflow as tf
import imageio
import PIL
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import *
from load import load_dataset

def vae(params):
    x_train, x_val, x_test = load_dataset(params)

    # specify the optimizer
    optimizer = tf.keras.optimizers.Adam()

    # instanties the VAE model
    model = VAE(params)

    # training and evaluation
    train_model(params, x_train)
    evaluate_model(params, x_train, x_val)


def train_model(params, x_train):
    loss, re, kl_div = 0, 0, 0
    pass

def evaluate_model(params, x_train, x_val):
    pass


class VAE(tf.keras.Model):
    def __init__(self, params):
        super(VAE, self).__init__()
        self.params = params

        hidden_units = params['hidden-units']
        latent_units = params['latent-units']

        self.encoder = tf.keras.Sequential([
            tf.keras.Input(shape=(784,)),
            GatedDenseLayer(hidden_units),
            GatedDenseLayer(hidden_units)
        ])

        self.latent_mean = tf.keras.Sequential([
            tf.keras.Input(shape=(hidden_units,)),
            tf.keras.layers.Dense(latent_units)
        ])
        self.latent_logvar = tf.keras.Sequential([
            tf.keras.Input(shape=(hidden_units,)),
            tf.keras.layers.Dense(latent_units, activation=Hardtanh(-6., 2.))
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.Input(shape=(latent_units,)),
            GatedDenseLayer(hidden_units)
        ])
        self.decoder_mean = tf.keras.Sequential([
            tf.keras.Input(shape=(hidden_units,)),
            tf.keras.layers.Dense(784, activation=tf.nn.sigmoid)
        ])


        # TODO: finish pseudo inputs


    def loss(self, inputs):
        x_mean, x_logvar, latent, latent_mean, latent_logvar = self.forward_pass(inputs)
        pass

    def forward_pass(self, input):
        pass

    def prior(self, z):
        # for gaussian-prior
        log_norm = -0.5*tf.pow(z, 2)
        return tf.math.reduce_sum(log_norm, 1)

    def generate_x(self, z):
        pass

    def generative_dist(self, z):
        pass


class Hardtanh():
    """
    Hardtangh implementation
    """
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.slope = tf.constant((max_val-min_val)/2., 'float32')
        self.offset = tf.constant((min_val+max_val)/2., 'float32')

    def __call__(self, x):
        x = tf.multiply(x, self.slope)
        x = tf.add(x, self.offset)
        x = tf.clip_by_value(x, self.min_val, self.max_val)
        return x


# Not used, example from tensorflow pages
class LinearLayer(Layer):

    def __init__(self, units):
        super(LinearLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.w) + self.b


class GatedDenseLayer(Layer):

    def __init__(self, units, name=None, activation=None):
        super(GatedDenseLayer, self).__init__(name=name)
        self.units = units
        self.activation = activation
        self.h = Dense(units)
        self.g = Dense(units)

    def build(self, input_shape):
        self.h.build(input_shape)
        self.g.build(input_shape)

    def call(self, inputs, **kwargs):
        h = self.h(inputs)
        if self.activation is not None:
            h = self.activation(self.h(inputs))

        g = tf.nn.sigmoid(inputs)
        return h #tf.math.multiply(h, g) <<-- OVO TREBA SKUZIT KAKO NAPRAVITI ???
