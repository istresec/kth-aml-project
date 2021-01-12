from utils.dataset_loading import load_dataset
from models.VAE import VAE


def test_vae(params):
    x_train, x_val, x_test = load_dataset(params)

    # instantiate the VAE model
    model = VAE(params)

    # training and evaluation
    model.train(x_train, x_val)
    model.evaluate_model(x_train, x_val)
