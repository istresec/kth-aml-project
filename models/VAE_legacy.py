import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import *
import time

from utils.distributions import *
from utils.Hardtanh import Hardtanh
from models.BaseModel import BaseModel
from utils.GatedDenseLayer import GatedDenseLayer
from utils.util import reparameterize


class VAE(BaseModel):
    """
    Classic 1-layer VAE implementation.
    """

    def __init__(self, params):
        super(VAE, self).__init__(params)
        self.params = params

        hidden_units = params['hidden-units']
        latent_units = params['latent-units']
        input_size = params['input-size']

        self.encoder = tf.keras.Sequential([
            tf.keras.Input(shape=(input_size,)),
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

        self.reparameterization = Lambda(reparameterize)

        self.decoder = tf.keras.Sequential([
            tf.keras.Input(shape=(latent_units,)),
            GatedDenseLayer(hidden_units),
            GatedDenseLayer(hidden_units)
        ])
        self.decoder_mean = tf.keras.Sequential([
            tf.keras.Input(shape=(hidden_units,)),
            tf.keras.layers.Dense(input_size, activation=tf.nn.sigmoid)
        ])
        self.decoder_logvar = tf.keras.Sequential([
            tf.keras.Input(shape=(hidden_units,)),
            tf.keras.layers.Dense(input_size, activation=Hardtanh(-4.5, 0.))
        ])

        self.optimizer = tf.keras.optimizers.Adam(params['learning-rate'])

    def get_weights(self):
        """
        Gets all weights (all trainable variables).
        :return: All network weights.
        """
        weights = self.encoder.trainable_variables
        weights += self.latent_mean.trainable_variables
        weights += self.latent_logvar.trainable_variables
        weights += self.reparameterization.trainable_variables
        weights += self.decoder.trainable_variables
        weights += self.decoder_mean.trainable_variables
        weights += self.decoder_logvar.trainable_variables

        if self.params['prior'] == 'vampprior':
            weights += self.means.trainable_variables

        return weights

    # TODO: possibly plot loss or ELBO of training and validation over epochs
    def train(self, dataset, validation_dataset):
        for epoch in range(self.params['epochs']):
            start = time.time()

            for x in dataset:
                loss = self.train_step(x)

            print('Epoch {} took {} sec'.format(epoch + 1, time.time() - start))

            if epoch % 10 == 0:
                self.generate_and_save_images(epoch + 1)
                self.evaluate_model(dataset, validation_dataset)

        self.generate_and_save_images(self.params['epochs'] + 1)

        self.plot_latent(20)

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as self.tape:
            loss, reconstruction_loss, kl_loss = self.loss(x, training=True)

        # update weights using GradientTape (automatic differentiation of loss)
        trainable_variables = self.get_weights()
        grad = self.tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grad, trainable_variables))

    def forward_pass(self, x, training=False):
        z_dist_encode = self.encoder(x, training=training)
        z_mean = self.latent_mean(z_dist_encode, training=training)
        z_logvar = self.latent_logvar(z_dist_encode, training=training)

        # reparameterize
        z = self.reparameterization([z_mean, z_logvar], training=training)

        x_decode = self.decoder(z, training=training)
        x_mean = self.decoder_mean(x_decode, training=training)
        x_logvar = self.decoder_logvar(x_decode, training=training)

        return z_mean, z_logvar, z, x_mean, x_logvar

    @tf.function
    def loss(self, x, beta=1., training=False, average_loss=False):
        """
        Calculates the loss, equal to the negative ELBO.

        :param x: Data.
        :param beta: Regularization coefficient, 1 by default.
        :param training: Determines if the calculations will be used for training. False by default.
        :param average_loss: Determines if the calculated loss is averaged or not. False (not averaged) by default.
        :return: The loss (negative ELBO).
        """

        z_mean, z_logvar, z, x_mean, x_logvar = self.forward_pass(x, training=training)

        reconstruction_loss = self.recon_loss(x, x_mean, x_logvar)
        kl_loss = self.kl_loss(z, z_mean, z_logvar, training=training)
        loss = -reconstruction_loss + beta * kl_loss

        if average_loss:
            loss = tf.reduce_mean(loss)
            reconstruction_loss = tf.reduce_mean(reconstruction_loss)
            kl_loss = tf.reduce_mean(kl_loss)

        return loss, reconstruction_loss, kl_loss

    def kl_loss(self, z, z_mean, z_logvar, training=False):
        """
        Calculates the Monte Carlo estimate of the Kullback-Leibler loss
        (the negative MC estimate of the KL term in the ELBO).

        :param z: A sample of the latent variables for which the KL loss is calculated.
        :param z_mean: Means of the latent variables z.
        :param z_logvar: Logarithm of the variance of the latent variables z.
        :param training: Determines if the calculations will be used for training. False by default.
        :return: The KL loss term.
        """
        log_p_z = self.prior(z, training=training)
        log_q_z = log_normal_diag(z, z_mean, z_logvar, dim=1)
        kl_loss = -(log_p_z - log_q_z)

        return kl_loss

    def recon_loss(self, x, x_mean, x_logvar):
        """
        Calculates the Monte Carlo estimate of the reconstruction loss (negative reconstruction term of the ELBO).
        :param x:
        :param x_mean:
        :param x_logvar:
        :return:
        """
        reconstruction_loss = -log_logistic(x, x_mean, x_logvar, dim=1)

        return reconstruction_loss

    def prior(self, z, training=False):

        if self.params['prior'] == 'sg':
            log_norm = -0.5 * tf.pow(z, 2)
            return tf.math.reduce_sum(log_norm, 1)

        elif self.params['prior'] == 'vampprior':
            c = self.params['vamp-components']

            # calculate parameters
            x = self.means(self.default_input, training=training)

            # calculate parameters for given data
            z_mean = self.latent_mean(x, training=training)
            z_logvar = self.latent_logvar(x, training=training)

            # expand dim
            z = tf.expand_dims(z, 1)
            z_mean = tf.expand_dims(z_mean, 0)
            z_logvar = tf.expand_dims(z_logvar, 0)

            a = log_normal_diag(z, z_mean, z_logvar, dim=2) - tf.math.log(tf.cast(c, tf.float32))
            a_max = tf.math.reduce_max(a, 1)

            return a_max + tf.math.log(tf.reduce_sum(tf.math.exp(a - tf.expand_dims(a_max, 1)), 1))

    def evaluate_model(self, x_train, x_val):
        train_loss = tf.keras.metrics.Mean()
        val_loss = tf.keras.metrics.Mean()

        for x in x_train:
            train_loss(self.loss(x, training=False)[0])

        for x in x_val:
            val_loss(self.loss(x, training=False)[0])

        print(f'Training ELBO: {-train_loss.result()}, validation ELBO: {-val_loss.result()}.')

    def generate_x(self, n=16):
        z_sample_rand = 0
        if self.params['prior'] == 'sg':
            z_sample_rand = tf.random.normal([n, self.params['latent-units']])
        elif self.params['prior'] == 'vampprior':
            means = self.means(self.default_input)[0:n]
            z_sample_mean = self.latent_mean(means)
            z_sample_logvar = self.latent_logvar(means)

            # reparameterize
            z_sample_rand = self.reparameterization([z_sample_mean, z_sample_logvar], training=False)

        samples_rand = self.decoder_mean(self.decoder(z_sample_rand))

        return samples_rand

    def generate_and_save_images(self, epoch):
        # _, _, _, predictions = self.forward_pass(self.seeded_input, training=False)
        # gen = self.decoder(latent, training=False)
        gen = self.generate_x(16)
        gen = tf.reshape(gen, (16, 28, 28, 1))

        fig = plt.figure(figsize=(4, 4))

        for i in range(gen.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(gen[i, :, :, 0] * 255, cmap='gray')
            plt.axis('off')

        plt.savefig('images/vae_' + self.params['prior'] + '_image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()

    # TODO: remove if useless (designed for 2 latent variables, SG prior and MNIST only)
    def plot_latent(self, n, digit_size=28):
        # https://www.tensorflow.org/tutorials/generative/cvae
        """
        Plots a 2D manifolds of digits from the latent space.
        Not executed if there are less or more than 2 latent units/variables.
        :param n: Number of values per latent variable, final image is n x n digits.
        :param digit_size: Square dimension of each digit.
        """
        if self.params['latent-units'] != 2:
            print('Skipping plotting the 2D manifold, there are more (or less) than 2 latent variables.')
            return

        norm = tfp.distributions.Normal(0, 1)
        grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
        grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
        image_width = digit_size * n
        image_height = image_width
        image = np.zeros((image_height, image_width))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z = np.array([[xi, yi]])
                x_decoded = self.decoder_mean(self.decoder(z))
                digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
                image[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit.numpy()

        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='Greys_r')
        plt.axis('Off')
        plt.show()
