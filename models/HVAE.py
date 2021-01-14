import tensorflow as tf
from functools import reduce

from utils.GatedDenseLayer import GatedDenseLayer
from utils.Hardtanh import Hardtanh
from utils.util import log_normal_pdf


class HVAE(tf.keras.Model):
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
        self.z1_dim = config["z1-dim"]
        self.z2_dim = config["z2-dim"]
        self.hidden_dim = config["hidden-dim"]
        self.input_shape_ = config["input-shape"]
        input_element_length = reduce(lambda x, y: x * y, self.input_shape_)

        # q (z2 | x) - encoder
        self.q_z2 = tf.keras.Sequential([
            tf.keras.Input(shape=self.input_shape_),
            tf.keras.layers.Flatten(),
            GatedDenseLayer(self.hidden_dim),
            GatedDenseLayer(self.hidden_dim)
        ])

        self.q_z2_mean = tf.keras.Sequential([
            tf.keras.Input(shape=(self.hidden_dim,)),
            tf.keras.layers.Dense(self.z2_dim)
        ])
        self.q_z2_logvar = tf.keras.Sequential([
            tf.keras.Input(shape=(self.hidden_dim,)),
            tf.keras.layers.Dense(self.z2_dim, activation=Hardtanh(-6., 2.))
        ])

        # q (z1 | x, z2) - encoder
        self.q_z1_x = tf.keras.Sequential([
            tf.keras.Input(shape=self.input_shape_),
            tf.keras.layers.Flatten(),
            GatedDenseLayer(self.hidden_dim)
        ])
        self.q_z1_z2 = tf.keras.Sequential([
            tf.keras.Input(shape=(self.z2_dim,)),
            GatedDenseLayer(self.hidden_dim)
        ])
        self.q_z1_joint = tf.keras.Sequential([
            tf.keras.Input(shape=(2 * self.hidden_dim,)),
            GatedDenseLayer(self.hidden_dim)
        ])

        self.q_z1_mean = tf.keras.Sequential([
            tf.keras.Input(shape=(self.hidden_dim,)),
            tf.keras.layers.Dense(self.z1_dim)
        ])
        self.q_z1_logvar = tf.keras.Sequential([
            tf.keras.Input(shape=(self.hidden_dim,)),
            tf.keras.layers.Dense(self.z1_dim, activation=Hardtanh(-6., 2.))
        ])

        # decoder: p(z1 | z2)
        self.p_z1 = tf.keras.Sequential([
            tf.keras.Input(shape=(self.z2_dim,)),
            GatedDenseLayer(self.hidden_dim),
            GatedDenseLayer(self.hidden_dim)
        ])

        self.p_z1_mean = tf.keras.Sequential([
            tf.keras.Input(shape=(self.hidden_dim,)),
            tf.keras.layers.Dense(self.z1_dim)
        ])
        self.p_z1_logvar = tf.keras.Sequential([
            tf.keras.Input(shape=(self.hidden_dim,)),
            tf.keras.layers.Dense(self.z1_dim, activation=Hardtanh(-6., 2.))
        ])

        # decoder: p(x | z1, z2)
        self.p_x_z1 = tf.keras.Sequential([
            tf.keras.Input(shape=(self.z1_dim,)),
            GatedDenseLayer(self.hidden_dim)
        ])
        self.p_x_z2 = tf.keras.Sequential([
            tf.keras.Input(shape=(self.z2_dim,)),
            GatedDenseLayer(self.hidden_dim)
        ])
        self.p_x_joint = tf.keras.Sequential([
            tf.keras.Input(shape=(2 * self.hidden_dim,)),
            GatedDenseLayer(self.hidden_dim)
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

    def prior(self, z2):
        if self.config['prior'] == 'sg':
            log_z2 = log_normal_pdf(z2, 0., 0.)

            return log_z2

        elif self.config['prior'] == 'vampprior':
            c = tf.cast(self.components, tf.float32)
            pseudo_inputs = self.means(self.idle_input)
            z2_mean, z2_logvar = self.encode_z2(pseudo_inputs)

            z2 = tf.expand_dims(z2, 1)
            z2_mean = tf.expand_dims(z2_mean, 0)
            z2_logvar = tf.expand_dims(z2_logvar, 0)

            a = log_normal_pdf(z2, z2_mean, z2_logvar, raxis=2) - tf.math.log(c)
            a_max = tf.math.reduce_max(a, 1)

            log_z2 = a_max + tf.math.log(tf.reduce_sum(tf.math.exp(a - tf.expand_dims(a_max, 1)), 1))

            return log_z2

    def encode(self, x):
        # z2 ~ q(z2 | x)
        z2_mean, z2_logvar = self.encode_z2(x)
        z2 = self.reparametrize(z2_mean, z2_logvar)

        # z1 ~ q(z1 | x, z2)
        z1_mean, z1_logvar = self.encode_z1(x, z2)
        z1 = self.reparametrize(z1_mean, z1_logvar)

        return z2, z2_mean, z2_logvar, z1, z1_mean, z1_logvar

    def reparametrize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z1, z2, apply_sigmoid=False):
        # p(z1 | z2)
        z1_mean, z1_logvar = self.decode_z1(z2)

        # x_mean = p(x | z1, z2)
        x_mean, x_logvar = self.decode_x(z1, z2, apply_sigmoid)

        return x_mean, x_logvar, z1_mean, z1_logvar

    def encode_z2(self, x):
        z2_dist_encode = self.q_z2(x)
        z2_mean = self.q_z2_mean(z2_dist_encode)
        z2_logvar = self.q_z2_logvar(z2_dist_encode)
        return z2_mean, z2_logvar

    def encode_z1(self, x, z2):
        x_enc = self.q_z1_x(x)
        z2_enc = self.q_z1_z2(z2)
        joint_input = tf.concat([x_enc, z2_enc], 1)
        z1_dist_encode = self.q_z1_joint(joint_input)
        z1_mean = self.q_z1_mean(z1_dist_encode)
        z1_logvar = self.q_z1_logvar(z1_dist_encode)
        return z1_mean, z1_logvar

    def decode_z1(self, z2):
        z1 = self.p_z1(z2)
        z1_mean = self.p_z1_mean(z1)
        z1_logvar = self.p_z1_logvar(z1)
        return z1_mean, z1_logvar

    def decode_x(self, z1, z2, apply_sigmoid=False):
        z1 = self.p_x_z1(z1)
        z2 = self.p_x_z2(z2)
        joint_input = tf.concat((z1, z2), 1)
        hidden = self.p_x_joint(joint_input)

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
                z2_sample = tf.random.normal([n, self.z2_dim])
            else:
                z2_sample = self.reparametrize(*self.encode(test_sample))

        elif self.config['prior'] == 'vampprior':
            n_pseudo_inputs = self.means(self.idle_input)[0:n]
            z2_sample_mean, z2_sample_logvar = self.encode_z2(n_pseudo_inputs)
            z2_sample = self.reparametrize(z2_sample_mean, z2_sample_logvar)

        z1_sample_mean, z1_sample_logvar = self.decode_z1(z2_sample)
        z1_sample = self.reparametrize(z1_sample_mean, z1_sample_logvar)

        x_mean, _ = self.decode_x(z1_sample, z2_sample, apply_sigmoid=True)

        return x_mean


def compute_loss(model, x):
    """
    For computing loss of the HVAE model.
    :param model: HVAE model.
    :param x: Current batch of data.
    :return: The loss.
    """
    # encode
    z2_q, z2_q_mean, z2_q_logvar, z1_q, z1_q_mean, z1_q_logvar = model.encode(x)

    # decode
    x_mean, x_logvar, z1_p_mean, z1_p_logvar = model.decode(z1_q, z2_q)

    # RE
    if model.config["x-variable-type"] == "binary":
        reduce_dims = tf.range(1, tf.rank(x))  # for example, returns [1, 2, 3] for dataset of 4 dimensional elements
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_mean, labels=x)
        log_p_x_z = -tf.reduce_sum(cross_ent, axis=reduce_dims)
    elif model.config["x-variable-type"] == "continuous":
        raise Exception("not implemented") # TODO

    # KL
    log_p_z1 = log_normal_pdf(z1_q, z1_p_mean, z1_p_logvar)
    log_q_z1_x = log_normal_pdf(z1_q, z1_q_mean, z1_q_logvar)
    log_p_z2 = model.prior(z2_q)
    log_q_z2_x = log_normal_pdf(z2_q, z2_q_mean, z2_q_logvar)

    return -tf.reduce_mean(log_p_x_z + log_p_z1 + log_p_z2 - log_q_z1_x - log_q_z2_x)
