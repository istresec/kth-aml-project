import numpy as np
import os
import tensorflow as tf


def load_dataset(params):
    """
    Function which loads chosen dataset
    :param params: parameter dictionary
    :return: train, validation and test tensor datasets
    """
    loader = loaders[params['dataset']]
    return loader(params)


def mnist_static_load(params):
    """
    Loads mnist dataset from files
    :param params: parameter dictioinary
    :return: tensor objects of train, validation and test datasets
    """

    seed = int(params['seed'])
    batch_size = int(params['batch-size'])

    convert = lambda lines: np.array([[int(num) for num in line.split()] for line in lines])

    with open(os.path.join('datasets', 'mnist', 'binarized_mnist_train.amat')) as file:
        lines = file.readlines()
    x_train = convert(lines).astype('float32')
    with open(os.path.join('datasets', 'mnist', 'binarized_mnist_valid.amat')) as file:
        lines = file.readlines()
    x_val = convert(lines).astype('float32')
    with open(os.path.join('datasets', 'mnist', 'binarized_mnist_test.amat')) as file:
        lines = file.readlines()
    x_test = convert(lines).astype('float32')

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024, seed=seed).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
    val_dataset = val_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, val_dataset, test_dataset


def mnist_load(params):
    """
    Loads mnist dataset
    :param params: params dictionary containing seed and batch size
    :return: tensor objects of train, validation and test datasets
    """

    seed = int(params['seed'])
    batch_size = int(params['batch-size'])

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, 784)) / 255.
    x_test = np.reshape(x_test, (-1, 784)) / 255.

    # get the binarized representation
    x_train = np.random.binomial(1, x_train)
    x_test = np.random.binomial(1, x_test)

    # Use 10000 samples for validation
    x_val = x_train[-10000:]
    x_train = x_train[:-10000]

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024, seed=seed).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
    val_dataset = val_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, val_dataset, test_dataset


loaders = {
    'mnist': mnist_load,
    'mnist-static': mnist_static_load
}
