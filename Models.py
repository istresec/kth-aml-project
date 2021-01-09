import tensorflow as tf
import imageio
import PIL
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import *
from load import load_dataset
from distributions import *
import time

def sampling_normal(inputs):
    mean, logvar = inputs
    eps = tf.keras.backend.random_normal(shape=tf.shape(mean))
    return mean + tf.exp(0.5 * logvar) * eps

def vae(params):
    x_train, x_val, x_test = load_dataset(params)

    # instanties the VAE model
    model = VAE(params)

    # training and evaluation
    model.train(x_train)
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

        # TODO: don't hardcode input size
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

        self.sample = Lambda(sampling_normal)

        self.decoder = tf.keras.Sequential([
            tf.keras.Input(shape=(latent_units,)),
            GatedDenseLayer(hidden_units),
            GatedDenseLayer(hidden_units)
        ])
        self.decoder_mean = tf.keras.Sequential([
            tf.keras.Input(shape=(hidden_units,)),
            tf.keras.layers.Dense(784, activation=tf.nn.sigmoid)
        ])
        self.decoder_logvar = tf.keras.Sequential([
            tf.keras.Input(shape=(hidden_units,)),
            tf.keras.layers.Dense(784, activation=Hardtanh(-4.5, 0.))
        ])

        self.optimizer = tf.keras.optimizers.Adam(params['learning-rate'])

        # TODO: finish pseudo inputs

    def get_weights(self):
        weights = self.encoder.trainable_variables
        weights += self.latent_mean.trainable_variables
        weights += self.latent_logvar.trainable_variables
        weights += self.sample.trainable_variables
        weights += self.decoder.trainable_variables
        weights += self.decoder_mean.trainable_variables
        weights += self.decoder_logvar.trainable_variables

        return weights

    @tf.function
    def loss(self, x, beta=1., average=False):
        z_mean, z_logvar, z, x_mean, x_logvar = self.forward_pass(x, training=True)

        reconstruction_loss = self.calc_reconstruction_loss(x, x_mean, x_logvar)
        kl_loss = self.calc_kl_loss(z_mean, z_logvar, z)
        loss = -reconstruction_loss + beta*kl_loss

        if average:
            loss = tf.reduce_mean(loss)
            reconstruction_loss = tf.reduce_mean(reconstruction_loss)
            kl_loss = tf.reduce_mean(kl_loss)

        return loss, reconstruction_loss, kl_loss

    @tf.function
    def forward_pass(self, x, training=False):
        z_mean = self.latent_mean(self.encoder(x, training=training), training=training)
        z_logvar = self.latent_logvar(self.encoder(x, training=training), training=training)
        z = self.sample([z_mean, z_logvar], training=training)

        x_decode = self.decoder(z, training=training)
        x_mean = self.decoder_mean(x_decode, training=training)
        x_logvar = self.decoder_logvar(x_decode, training=training)

        return z_mean, z_logvar, z, x_mean, x_logvar

    def prior(self, z):
        # for gaussian-prior
        log_norm = -0.5*tf.pow(z, 2)
        return tf.math.reduce_sum(log_norm, 1)

    def train(self, dataset):
        for epoch in range(self.params['epochs']):
            start = time.time()

            for x in dataset:
                self.train_step(x)

            if epoch % 10 == 0:
                self.generate_and_save_images(epoch+1)

            print('Epoch {} took {} sec'.format(epoch + 1, time.time()-start))

        self.generate_and_save_images(self.params['epochs']+1)

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as self.tape:
            # TODO: use loss method - didn't get it to work proerly
            # forward pass
            z_mean, z_logvar, z, x_mean, x_logvar = self.forward_pass(x, training=True)

            # calculate loss
            reconstruction_loss = self.calc_reconstruction_loss_temp(x, x_mean, x_logvar)
            kl_loss = self.calc_kl_loss_temp(z_mean, z_logvar, z)
            loss = reconstruction_loss + kl_loss

        # train weights using calculated loss and given gradient
        trainable_variables = self.get_weights()
        grad = self.tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grad, trainable_variables))

    def calc_kl_loss(self, z_mean, z_logvar, z):
        log_p_z = self.prior(z)
        log_q_z = log_Normal_diag(z, z_mean, z_logvar, dim=1)
        kl_loss = -(log_p_z - log_q_z)

        return kl_loss

    def calc_kl_loss_temp(self, z_mean, z_logvar, z):
        # Placeholder function for testing
        kl_loss = 1 + z_logvar - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_logvar)
        kl_loss = -0.5 * tf.keras.backend.sum(kl_loss, axis=-1)

        return kl_loss

    def calc_reconstruction_loss(self, x, x_mean, x_logvar):
        reconstruction_loss = -log_Logistic(x, x_mean, x_logvar, dim=1)

        return reconstruction_loss

    def calc_reconstruction_loss_temp(self, x, x_mean, x_logvar):
        # Placeholder function for testing
        reconstruction_loss = 784*tf.keras.losses.binary_crossentropy(x, x_mean)

        return reconstruction_loss

    def generate_x(self, N=16):
        z_sample_rand = 0
        if self.params['prior'] == 'gaussian':
            z_sample_rand = tf.random.normal([N, self.params['latent-units']])
        elif self.params['prior'] == 'vampprior':
            pass
        samples_rand = self.decoder_mean(self.decoder(z_sample_rand))

        return samples_rand

    def generative_dist(self, z):
        pass

    def generate_and_save_images(self, epoch):
        # _, _, _, predictions = self.forward_pass(self.seeded_input, training=False)
        # gen = self.decoder(latent, training=False)
        gen = self.generate_x(16)
        gen = tf.reshape(gen, (16, 28, 28, 1))

        fig = plt.figure(figsize=(4,4))

        for i in range(gen.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(gen[i, :, :, 0] * 255, cmap='gray')
            plt.axis('off')

        plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()

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
