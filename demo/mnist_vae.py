import numpy as np
import os
import tensorflow as tf

from models.VAE import compute_loss, VAE
from trainers.vae_trainer import VAETrainer
from utils.util import project_path, ensure_dirs, load_mnist

if __name__ == '__main__':
    tf.random.set_seed(72)
    np.random.seed(72)

    run_name = "vae_sg_01"

    config = dict()
    config['run_name'] = run_name
    config['hidden_dim'] = 300
    config['latent_dim'] = 40
    config['epochs'] = 100
    config['batch_size'] = 128
    config['logging_interval'] = 1
    config['summary_dir'] = os.path.join(project_path, f"tb/{run_name}")
    config['checkpoint_dir'] = os.path.join(project_path, f"checkpoints/{run_name}")

    config['prior'] = 'vampprior'  # or sg
    config['vamp_components'] = 500

    ensure_dirs([config["summary_dir"], config["checkpoint_dir"]])

    train_dataset, valid_dataset = load_mnist(config["batch_size"])
    config['input_shape'] = tuple(train_dataset.element_spec.shape)[1:]

    optimizer = tf.keras.optimizers.Adam(1e-4)
    model = VAE(config)
    loss_function = compute_loss
    writer = tf.summary.create_file_writer(config["summary_dir"])

    trainer = VAETrainer(optimizer, model, train_dataset, valid_dataset, loss_function, config, writer)
    trainer.train()
