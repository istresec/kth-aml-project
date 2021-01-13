import numpy as np
import os
import tensorflow as tf

from tests import test_vae

if __name__ == '__main__':
    if not os.path.exists('images'):
        os.mkdir('images')

    # setup reproducibility
    tf.random.set_seed(72)
    np.random.seed(72)

    # set parameters
    params = dict()
    params['epochs'] = 100
    params['hidden-units'] = 300
    params['batch-size'] = 128
    params['seed'] = 97
    params['prior'] = 'sg'  # {'sg', 'vampprior'}
    params['vamp-components'] = 500
    params['dataset'] = 'mnist'  # or 'mnist-static'
    params['input-size'] = 784  # depends on the dataset, 784 for MNIST
    params['latent-units'] = 40  # suggested: 40
    params['learning-rate'] = 1e-4

    # experiments
    test_vae(params)
