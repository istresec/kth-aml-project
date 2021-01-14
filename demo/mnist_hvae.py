import numpy as np
import os
import tensorflow as tf

from models.HVAE import compute_loss, HVAE
from trainers.vae_trainer import VAETrainer
from utils.util import project_path, ensure_dirs, get_str_formatted_time, load_mnist

if __name__ == '__main__':
    config = dict()
    config['hidden-dim'] = 300
    config['z1-dim'] = 40
    config['z2-dim'] = 40
    config['epochs'] = 300
    config['batch-size'] = 128
    config['logging-interval'] = 1
    config["seed"] = 72
    config['prior'] = 'vampprior'  # {'vampprior', 'sg'}
    config['vamp-components'] = 500

    run_name = f"HVAE_prior-{config['prior']}_z1-{config['z1-dim']}_z2-{config['z2-dim']}_bs-{config['batch-size']}_____{get_str_formatted_time()}"
    config['run-name'] = run_name
    config['summary-dir'] = os.path.join(project_path, f"tb/{run_name}")
    config['images-dir'] = os.path.join(project_path, f"images/{run_name}")
    config['checkpoint-dir'] = os.path.join(project_path, f"checkpoints/{run_name}")

    tf.random.set_seed(config["seed"])
    np.random.seed(config["seed"])
    ensure_dirs([config["summary-dir"], config["checkpoint-dir"], config['images-dir']])

    train_dataset, valid_dataset = load_mnist(config["batch-size"])
    # train_dataset, valid_dataset, test_dataset = loaders["mnist"](config)
    config['input-shape'] = tuple(train_dataset.element_spec.shape)[1:]
    print("Dataset loaded.")
    print(f"Config:{config}")

    optimizer = tf.keras.optimizers.Adam(1e-4)
    model = HVAE(config)
    loss_function = compute_loss
    writer = tf.summary.create_file_writer(config["summary-dir"])

    trainer = VAETrainer(optimizer, model, train_dataset, valid_dataset, loss_function, config, writer)
    trainer.train()
