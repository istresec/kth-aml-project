from tests import test_vae
import os

if __name__ == '__main__':
    if not os.path.exists('images'):
        os.mkdir('images')

    # set parameters
    params = dict()
    params['epochs'] = 100
    params['hidden-units'] = 300
    params['batch-size'] = 100
    params['seed'] = 97
    params['prior'] = 'vampprior'  # or 'sg'
    params['vamp-components'] = 500
    params['dataset'] = 'mnist'  # or 'mnist-static'
    params['input-size'] = 784  # depends on the dataset, 784 for MNIST
    params['latent-units'] = 40  # suggested: 40
    params['learning-rate'] = 1e-4

    # experiments
    test_vae(params)
