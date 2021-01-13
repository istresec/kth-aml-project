import numpy as np
import tensorflow as tf

from utils.GatedDenseLayer import GatedDenseLayer
from utils.Hardtanh import Hardtanh


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, parameters):
        super().__init__(parameters)

        self.latent_dim = parameters["latent_dim"]

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation="relu"),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation="relu"),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(7 * 7 * 32, tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same",
                                                activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same",
                                                activation="relu"),
                # No activation
                tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding="same")
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparametrize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


class VAE(tf.keras.Model):
    def __init__(self, parameters):
        super().__init__(parameters)

        self.latent_dim = parameters["latent_dim"]
        self.hidden_dim = parameters["hidden_dim"]

        self.encoder = tf.keras.Sequential([
            tf.keras.Input(shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            GatedDenseLayer(self.hidden_dim),
            GatedDenseLayer(self.hidden_dim)
        ])
        self.latent_mean = tf.keras.Sequential([
            tf.keras.Input(shape=(self.hidden_dim,)),
            tf.keras.layers.Dense(self.latent_dim)
        ])
        self.latent_logvar = tf.keras.Sequential([
            tf.keras.Input(shape=(self.hidden_dim,)),
            tf.keras.layers.Dense(self.latent_dim, activation=Hardtanh(-6., 2.))
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.Input(shape=(self.latent_dim,)),
            GatedDenseLayer(self.hidden_dim),
            GatedDenseLayer(self.hidden_dim),
            tf.keras.layers.Dense(28 * 28),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1))
        ])

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        z_dist_encode = self.encoder(x)
        mean = self.latent_mean(z_dist_encode)
        logvar = self.latent_logvar(z_dist_encode)
        return mean, logvar

    def reparametrize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparametrize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logz - logqz_x)
