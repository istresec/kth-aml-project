import io
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time

from models.VAE import generate_4x4_images_grid


class VAETrainer:
    def __init__(self, optimizer, model, train_dataset, valid_dataset, loss_function, config, writer):
        self.optimizer = optimizer
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.loss_function = loss_function
        self.config = config

        self.writer = writer
        assert config["batch-size"] >= 16
        for batch in valid_dataset.take(1):
            self.log_test_sample = batch[0:16]

    def train(self):
        print("Training started")
        self.log_config()
        self.log_images(epoch=0)
        for epoch in range(1, self.config["epochs"] + 1):
            self.train_epoch(epoch)

    def train_epoch(self, epoch):
        summaries_dict = dict()

        start_time = time.time()
        train_loss = tf.keras.metrics.Mean()
        for train_x in self.train_dataset:
            train_loss(self.train_step(train_x))
        end_time = time.time()

        train_elbo = -train_loss.result()
        summaries_dict['train_elbo'] = train_elbo

        if (epoch - 1) % self.config["logging-interval"] == 0:
            valid_loss = tf.keras.metrics.Mean()
            for test_x in self.valid_dataset:
                valid_loss(self.train_step(test_x, training=False))

            valid_elbo = -valid_loss.result()
            summaries_dict['valid_elbo'] = valid_elbo

            self.log_images(epoch)
            with self.writer.as_default():
                for k, v in summaries_dict.items():
                    tf.summary.scalar(name=k, data=v, step=epoch)
                for i, v in enumerate(self.model.trainable_variables):
                    tf.summary.histogram(f'trainable_variables[{i}]', v, step=epoch)

        print(f"Epoch: {epoch}, time elapsed: {end_time - start_time}, summaries: {summaries_dict}")
        # generate_and_save_images(model, epoch, test_sample)

    @tf.function
    def train_step(self, x, training=True):
        if not training:
            return self.loss_function(self.model, x)

        with tf.GradientTape() as tape:
            loss = self.loss_function(self.model, x)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def log_config(self):
        with self.writer.as_default():
            with open(os.path.join(self.config["images-dir"], "config.txt"), "w") as f:
                f.write(f"{self.config}\n")

    def log_images(self, epoch):
        with self.writer.as_default():
            # TODO hardcoded 28x28. Maybe move to config
            generate_4x4_images_grid(self.model, epoch, (28, 28), self.log_test_sample)

            # 1. save png to images-dir
            plt.savefig(os.path.join(self.config['images-dir'], f"epoch={epoch:05d}"), format="png")

            # 2. add a tensorboard summary
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            tf_image = tf.expand_dims(tf.image.decode_png(buf.getvalue(), channels=4), 0)
            tf.summary.image("Training data", tf_image, step=epoch)
