import tensorflow as tf
import time


class VAETrainer:
    def __init__(self, optimizer, model, train_dataset, valid_dataset, loss_function, config, writer):
        self.optimizer = optimizer
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.loss_function = loss_function
        self.config = config
        self.writer = writer

    def train(self):
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

            with self.writer.as_default():
                for k, v in summaries_dict.items():
                    tf.summary.scalar(name=k, data=v, step=epoch)
                for i, v in enumerate(self.model.trainable_variables):
                    tf.summary.histogram(f'trainable_variables[{i}]', v, step=epoch)

            # TODO log images

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
