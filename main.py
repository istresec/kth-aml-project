import tensorflow as tf
import imageio
import PIL
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import *
from Models import vae



if __name__ == '__main__':

    # parameters
    params = dict()
    params['epochs'] = 100
    params['hidden-units'] = 300
    params['batch-size'] = 64
    params['seed'] = 97
    params['prior'] = 'gaussian' # or 'vampprior'
    params['dataset'] = 'mnist' # or 'mnist-static'
    params['latent-units'] = 40
    params['learning-rate'] = 1e-4


    vae(params)
