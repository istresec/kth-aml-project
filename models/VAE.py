import tensorflow as tf
from functools import reduce

from utils.GatedDenseLayer import GatedDenseLayer
from utils.Hardtanh import Hardtanh
from utils.util import log_normal_pdf


class VAE(tf.keras.Model):
    """
    Variational autoencoder with standard gaussian prior.

    Uses gated dense layers for both encoder and decoder.

    p(z) is the prior.
    p(x given z) is modeled by a bernoulli distribution.
    q(z given x) is modeled by a log-normal distribution.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.latent_dim = config["latent-dim"]
        self.hidden_dim = config["hidden-dim"]
        self.input_shape_ = config["input-shape"]
        input_element_length = reduce(lambda x, y: x * y, self.input_shape_)

        self.encoder = tf.keras.Sequential([
            tf.keras.Input(shape=self.input_shape_),
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
        ])
        if config["x-variable-type"] == "binary":
            self.p_x_mean = tf.keras.Sequential([
                tf.keras.Input(shape=(self.hidden_dim,)),
                tf.keras.layers.Dense(input_element_length),
                tf.keras.layers.Reshape(target_shape=self.input_shape_)
            ])
        elif config["x-variable-type"] == "continuous":
            self.p_x_mean = tf.keras.Sequential([
                tf.keras.Input(shape=(self.hidden_dim,)),
                tf.keras.layers.Dense(input_element_length),
                tf.keras.layers.Reshape(target_shape=self.input_shape_)
            ])
            self.p_x_logvar = tf.keras.Sequential([
                tf.keras.Input(shape=(self.hidden_dim,)),
                tf.keras.layers.Dense(input_element_length, activation=Hardtanh(-6., 2.)),
                tf.keras.layers.Reshape(target_shape=self.input_shape_)
            ])
        else:
            raise Exception(f"Invalid value given for 'x-variable-type': value was '{config['x-variable-type']}'")

        if self.config['prior'] == 'vampprior':
            self.components = self.config['vamp-components']

            self.means = tf.keras.Sequential([
                tf.keras.Input(shape=(self.components,)),
                tf.keras.layers.Dense(input_element_length, activation=Hardtanh(0., 1.)),
                tf.keras.layers.Reshape(target_shape=self.input_shape_)
            ])

            self.idle_input = tf.Variable(tf.eye(self.components, self.components, dtype=tf.float32))

    def prior(self, z):
        if self.config['prior'] == 'sg':
            logz = log_normal_pdf(z, 0., 0.)

            return logz

        elif self.config['prior'] == 'vampprior':
            c = tf.cast(self.components, tf.float32)
            pseudo_inputs = self.means(self.idle_input)
            mean, logvar = self.encode(pseudo_inputs)

            z = tf.expand_dims(z, 1)
            mean = tf.expand_dims(mean, 0)
            logvar = tf.expand_dims(logvar, 0)

            a = log_normal_pdf(z, mean, logvar, raxis=2) - tf.math.log(c)
            a_max = tf.math.reduce_max(a, 1)

            logz = a_max + tf.math.log(tf.reduce_sum(tf.math.exp(a - tf.expand_dims(a_max, 1)), 1))

            return logz

    def encode(self, x):
        z_dist_encode = self.encoder(x)
        mean = self.latent_mean(z_dist_encode)
        logvar = self.latent_logvar(z_dist_encode)
        return mean, logvar

    def reparametrize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        hidden = self.decoder(z)
        if self.config["x-variable-type"] == "binary":
            logits = self.p_x_mean(hidden)
            if apply_sigmoid:
                probs = tf.sigmoid(logits)
                return probs, None
            return logits, None
        elif self.config["x-variable-type"] == "continuous":
            x_mean = self.p_x_mean(hidden)
            x_logvar = self.p_x_logvar(hidden)
            return x_mean, x_logvar

    def generate_x(self, n=1, test_sample=None):
        if self.config['prior'] == 'sg':
            if test_sample is not None:
                z_sample = tf.random.normal([n, self.latent_dim])
            else:
                z_sample = self.reparametrize(*self.encode(test_sample))

        elif self.config['prior'] == 'vampprior':
            n_pseudo_inputs = self.means(self.idle_input)[0:n]
            sample_mean, sample_logvar = self.encode(n_pseudo_inputs)
            z_sample = self.reparametrize(sample_mean, sample_logvar)

        samples_mean, _ = self.decode(z_sample, apply_sigmoid=True)

        return samples_mean


def compute_loss(model, x):
    """
    For computing loss of the VAE or CVAE model.
    :param model: VAE or CVAE model.
    :param x: Current batch of data.
    :return: The loss.
    """
    mean, logvar = model.encode(x)
    z = model.reparametrize(mean, logvar)
    x_mean, x_logvar = model.decode(z)

    if model.config["x-variable-type"] == "binary":
        reduce_dims = tf.range(1, tf.rank(x))  # for example, returns [1, 2, 3] for dataset of 4 dimensional elements
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_mean, labels=x)
    elif model.config["x-variable-type"] == "continuous":
        raise Exception("not impl")  # TODO

    logpx_z = -tf.reduce_sum(cross_ent, axis=reduce_dims)
    logz = model.prior(z)  # logz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)

    return -tf.reduce_mean(logpx_z + logz - logqz_x)
