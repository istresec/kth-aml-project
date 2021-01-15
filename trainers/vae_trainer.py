import io
import os
import time

import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.util import generate_images_grid


class VAETrainer:
    def __init__(self, optimizer, model, train_dataset, valid_dataset, loss_function, loglikelihood_function, config,
                 writer):
        self.optimizer = optimizer
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.loss_function = loss_function
        self.loglikelihood_function = loglikelihood_function
        self.config = config

        self.writer = writer
        assert config["batch-size"] >= config["log-images-grid-size"] ** 2
        for batch in valid_dataset.take(1):
            self.log_test_sample = batch[0:config["log-images-grid-size"] ** 2]

    def train(self):
        print("Training started")
        self.log_config()
        self.log_images(epoch=0, sample=self.log_test_sample, plot_predictions=False)
        self.log_images(epoch=0, sample=self.log_test_sample)
        for epoch in range(1, self.config["epochs"] + 1):
            self.train_epoch(epoch)

        print("Computing loglikelihood")
        valid_ll = tf.keras.metrics.Sum()
        val_dataset_for_ll = self.valid_dataset._input_dataset.batch(
            self.config["ll-batch-size"])  # TODO this is a hack
        for test_x in tqdm(val_dataset_for_ll):
            ll = self.loglikelihood_function(self.model, self.loss_function, test_x, self.config["ll-n-samples"])
            valid_ll(ll)
        valid_ll = valid_ll.result()

        with self.writer.as_default():
            tf.summary.scalar(name="valid_ll", data=valid_ll, step=epoch)
        print(f"Likelihood is: {valid_ll}")

    def train_epoch(self, epoch):
        print(f"Epoch {epoch} started.")
        summaries_dict = dict()

        start_time = time.time()
        train_loss = tf.keras.metrics.Mean()
        for train_x in tqdm(self.train_dataset):
            loss = self.train_step(train_x)
            train_loss(loss)
        end_time = time.time()

        train_elbo = -train_loss.result()
        summaries_dict['train_elbo'] = train_elbo

        if (epoch - 1) % self.config["logging-interval"] == 0:
            valid_loss = tf.keras.metrics.Mean()
            for test_x in self.valid_dataset:
                loss, _ = self.loss_function(self.model, test_x)
                valid_loss(loss)

            valid_elbo = -valid_loss.result()
            summaries_dict['valid_elbo'] = valid_elbo

            self.log_images(epoch, self.log_test_sample)
            with self.writer.as_default():
                for k, v in summaries_dict.items():
                    tf.summary.scalar(name=k, data=v, step=epoch)
                for i, v in enumerate(self.model.trainable_variables):
                    tf.summary.histogram(f'trainable_variables[{i}]', v, step=epoch)

        print(f"Epoch: {epoch}, time elapsed: {end_time - start_time}, summaries: {summaries_dict}")

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss, _ = self.loss_function(self.model, x)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def log_config(self):
        with self.writer.as_default():
            with open(os.path.join(self.config["images-dir"], "config.txt"), "w") as f:
                f.write(f"{self.config}\n")

    def log_images(self, epoch, sample, plot_predictions=True):
        with self.writer.as_default():
            generate_images_grid(
                self.model, epoch,
                self.config["log-images-grid-size"],
                self.config["log-images-shape"],
                sample, plot_predictions
            )

            # 1. save png to images-dir
            plt.savefig(os.path.join(self.config['images-dir'], f"epoch={epoch:05d}"), format="png")

            # 2. add a tensorboard summary
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            tf_image = tf.expand_dims(tf.image.decode_png(buf.getvalue(), channels=4), 0)
            tf.summary.image(f"{'Predictions' if plot_predictions else 'Real images'}", tf_image, step=epoch)
