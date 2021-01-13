import tensorflow as tf
from functools import reduce


class CVAE(tf.keras.Model):
    """
    Convolutional variational autoencoder with standard gaussian prior.

    Uses a simple convolution layers structure (conv+relu+conv+relu+flatten+dense) for encoder
    and a symmetric deconvolution structure for the decoder.

    p(z) is the prior.
    p(x given z) is modeled by a bernoulli distribution.
    q(z given x) is modeled by a log-normal distribution.
    """

    def __init__(self, parameters):
        super().__init__()

        self.latent_dim = parameters["latent_dim"]

        self.input_shape_ = parameters["input_shape"]
        input_element_length = reduce(lambda x, y: x * y, self.input_shape_)
        if input_element_length != 28 * 28:
            # TODO implement if needed. The CVAE wont work for other dimensions because
            #      the deconvolution is hardcoded to start from a (7x7x32) dense layer
            raise Exception("CVAE only implemented for images of 28x28 pixels.")

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.input_shape_),
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

    def generate_x(self, n=1):
        # TODO
        pass
