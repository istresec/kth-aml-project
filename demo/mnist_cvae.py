import numpy as np
import os
import tensorflow as tf

from models.CVAE import CVAE
from models.VAE import compute_loss
from trainers.vae_trainer import VAETrainer
from utils.util import project_path, ensure_dirs, load_mnist, get_str_formatted_time

if __name__ == '__main__':
    tf.random.set_seed(72)
    np.random.seed(72)

    config = dict()
    config['hidden-dim'] = 300
    config['latent-dim'] = 2
    config['epochs'] = 20
    config['batch-size'] = 32

    run_name = f"cvae_sg_01_{get_str_formatted_time()}"
    config['run_name'] = run_name
    config['logging-interval'] = 1
    config['summary-dir'] = os.path.join(project_path, f"tb/{run_name}")
    config['checkpoint-dir'] = os.path.join(project_path, f"checkpoints/{run_name}")

    ensure_dirs([config["summary-dir"], config["checkpoint-dir"]])

    train_dataset, valid_dataset = load_mnist(config["batch-size"])
    config['input-shape'] = tuple(train_dataset.element_spec.shape)[1:]

    optimizer = tf.keras.optimizers.Adam(1e-4)
    model = CVAE(config)
    loss_function = compute_loss
    writer = tf.summary.create_file_writer(config["summary-dir"])

    trainer = VAETrainer(optimizer, model, train_dataset, valid_dataset, loss_function, config, writer)
    trainer.train()
