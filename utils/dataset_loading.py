import numpy as np
import os
import tensorflow as tf
from scipy.io import loadmat

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

def fashion_mnist_load(params):
    """
    Loads fashion mnist dataset
    :param params: params dictionary containing seed and batch size
    :return: tensor objects of train, validation and test datasets
    """

    seed = int(params['seed'])
    batch_size = int(params['batch-size'])

    (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = np.reshape(x_train, (-1, 784)).astype('float32') / 255.
    x_test = np.reshape(x_test, (-1, 784)).astype('float32') / 255.

    x_val = x_train[-10000:]
    x_train = x_train[:-10000]

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024, seed=seed).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
    val_dataset = val_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, val_dataset, test_dataset

def omniglot_load(params):
    """
    Loads omniglot dataset
    :param params: params dictionary containing seed and batch size
    :return: tensor objects of train, validation and test datasets
    """

    seed = int(params['seed'])
    batch_size = int(params['batch-size'])

    data = loadmat(os.path.join('datasets', 'omniglot', 'chardata.mat'))
    data = data['data'].T.astype('float32').reshape(-1, 28, 28).reshape((-1, 28*28), order='F')

    train_data = data[:-4000]
    x_test = data[-4000:]

    np.random.shuffle(train_data)

    x_train = train_data[:-4000]
    x_val = train_data[-4000:]

    if params['binary']:
        x_train = np.random.binomial(1, x_train)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024, seed=seed).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
    val_dataset = val_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, val_dataset, test_dataset

def caltech101_load(params):
    """
    Loads caltech101 dataset
    :param params: params dictionary containing seed and batch size
    :return: tensor objects of train, validation and test datasets
    """

    seed = int(params['seed'])
    batch_size = int(params['batch-size'])

    data = loadmat(os.path.join('datasets', 'caltech101', 'caltech101_silhouettes_28_split1.mat'))

    x_train = 1. - data['train_data'].astype('float32').reshape(-1, 28, 28).reshape((-1, 28*28), order='F')
    np.random.shuffle(x_train)
    x_val = 1. - data['val_data'].astype('float32').reshape(-1, 28, 28).reshape((-1, 28*28), order='F')
    x_test = 1. - data['test_data'].astype('float32').reshape(-1, 28, 28).reshape((-1, 28*28), order='F')

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024, seed=seed).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
    val_dataset = val_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, val_dataset, test_dataset

loaders = {
    'mnist': mnist_load,
    'mnist-static': mnist_static_load,
    'fashion-mnist': fashion_mnist_load,
    'omniglot': omniglot_load,
    'caltech101': caltech101_load
}
