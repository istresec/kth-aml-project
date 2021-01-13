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

    def __init__(self, parameters):
        super().__init__()

        self.params = parameters
        self.latent_dim = parameters["latent_dim"]
        self.hidden_dim = parameters["hidden_dim"]
        self.input_shape_ = parameters["input_shape"]
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
            tf.keras.layers.Dense(input_element_length),
            tf.keras.layers.Reshape(target_shape=self.input_shape_)
        ])

        if self.params['prior'] == 'vampprior':
            self.components = self.params['vamp_components']

            self.means = tf.keras.Sequential([
                tf.keras.Input(shape=(self.components,)),
                tf.keras.layers.Dense(self.hidden_dim, activation=Hardtanh(0., 1.))
            ])

            self.idle_input = tf.Variable(tf.eye(self.components, self.components, dtype=tf.float32))

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def prior(self, z):
        if self.params['prior'] == 'sg':
            logz = log_normal_pdf(z, 0., 0.)

            return logz

        elif self.params['prior'] == 'vampprior':
            c = tf.cast(self.components, tf.float32)
            pseudo_inputs = self.means(self.idle_input)

            mean = self.latent_mean(pseudo_inputs)
            logvar = self.latent_logvar(pseudo_inputs)

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
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def generate_x(self, n=1):
        z_sample = 0

        if self.params['prior'] == 'sg':
            z_sample = tf.random.normal([n, self.latent_dim])
        elif self.params['prior'] == 'vampprior':
            n_pseudo_inputs = self.means(self.idle_input)[0:n]
            sample_mean = self.latent_mean(n_pseudo_inputs)
            sample_logvar = self.latent_logvar(n_pseudo_inputs)
            z_sample = self.reparametrize(sample_mean, sample_logvar)

        samples_rand = self.decoder_mean(self.decoder(z_sample))

        return samples_rand


def compute_loss(model, x):
    """
    For computing losses of both VAE_SG and CVAE_SG
    :param model:
    :param x: current batch
    :return: the loss
    """
    mean, logvar = model.encode(x)
    z = model.reparametrize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logz = model.prior(z)  # logz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logz - logqz_x)
