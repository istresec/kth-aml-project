import pathlib
from datetime import datetime

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

project_path = pathlib.Path(__file__).parent.parent


def reparameterize(inputs):
    mean, logvar = inputs
    eps = tf.keras.backend.random_normal(shape=tf.shape(mean))
    return mean + tf.exp(0.5 * logvar) * eps


def get_str_formatted_time() -> str:
    return datetime.now().strftime('%Y-%m-%d--%H.%M.%S')


def ensure_dir(dirname):
    dirname = pathlib.Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def ensure_dirs(dirs):
    for dir_ in dirs:
        ensure_dir(dir_)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


def load_mnist(batch_size):
    # TODO remove

    def preprocess_images(images):
        images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
        return np.where(images > .5, 1., 0.).astype("float32")

    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(len(train_images)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(len(test_images)).batch(batch_size)

    return train_dataset, test_dataset


def generate_images_grid(model, epoch, grid_size, image_shape, test_sample, plot_predictions=True):
    if plot_predictions:
        predictions = model.generate_x(grid_size ** 2, test_sample)
        images = tf.reshape(predictions, (-1, *image_shape))
    else:
        images = tf.reshape(test_sample, (-1, *image_shape))

    plt.figure(figsize=(grid_size, grid_size))
    plt.title(f"epoch:{epoch}")
    for i in range(images.shape[0]):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.axis("off")
