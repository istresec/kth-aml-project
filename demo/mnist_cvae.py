import numpy as np
import os
import tensorflow as tf

from models.VAE_2 import CVAE_SG, compute_loss
from trainers.vae2_trainer import VAE2Trainer
from utils.util import project_path, ensure_dirs, load_mnist

if __name__ == '__main__':
    tf.random.set_seed(72)
    np.random.seed(72)

    run_name = "cvae_sg_01"

    config = dict()
    config['run_name'] = run_name
    config['hidden_dim'] = 300
    config['latent_dim'] = 2
    config['epochs'] = 20
    config['batch_size'] = 32
    config['logging_interval'] = 1
    config['summary_dir'] = os.path.join(project_path, f"tb/{run_name}")
    config['checkpoint_dir'] = os.path.join(project_path, f"checkpoints/{run_name}")

    ensure_dirs([config["summary_dir"], config["checkpoint_dir"]])

    optimizer = tf.keras.optimizers.Adam(1e-4)
    model = CVAE_SG(config)
    train_dataset, valid_dataset = load_mnist(config["batch_size"])
    loss_function = compute_loss
    writer = tf.summary.create_file_writer(config["summary_dir"])

    trainer = VAE2Trainer(optimizer, model, train_dataset, valid_dataset, loss_function, config, writer)
    trainer.train()
