import os

from sklearn.utils import shuffle
import numpy as np
import pickle

from eoflow.base.configuration import Configurable

import tensorflow as tf


class BaseModelCustomTraining(tf.keras.Model, Configurable):
    def __init__(self, config_specs):
        tf.keras.Model.__init__(self)
        Configurable.__init__(self, config_specs)

        self.net = None
        self.init_model()

    def init_model(self):
        """Called on __init__. Keras self initialization. Create self here if does not require the inputs shape"""
        pass

    def build(self, inputs_shape):
        """Keras method. Called once to build the self. Build the self here if the input shape is required."""
        pass

    def call(self, inputs, training=False):
        pass

    def prepare(
        self,
        optimizer=None,
        loss=None,
        metrics=None,
        epoch_loss_metric=None,
        epoch_val_metric=None,
        reduce_lr=False,
        **kwargs,
    ):
        """Prepares the self for training and evaluation. This method should create the
        optimizer, loss and metric functions and call the compile method of the self. The self
        should provide the defaults for the optimizer, loss and metrics, which can be overriden
        with custom arguments."""

        raise NotImplementedError

    @tf.function
    def train_step(self, train_ds):

        for x_batch_train, y_batch_train in train_ds:
            with tf.GradientTape() as task_tape, tf.GradientTape() as enc_tape:

                # Forward pass
                x_enc = self.encoder(x_batch_train, training=True)
                y_preds = self.task(x_enc, training=True)

                cost = self.loss(y_batch_train, y_preds)
                # kernel regularizer
                cost += sum(self.encoder.losses) + sum(self.task.losses)

                cost = tf.reduce_mean(cost)

            gradients_enc = enc_tape.gradient(cost, self.encoder.trainable_variables)
            gradients_task = task_tape.gradient(cost, self.task.trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients_enc, self.encoder.trainable_variables)
            )
            self.optimizer.apply_gradients(
                zip(gradients_task, self.task.trainable_variables)
            )

            self.loss_metric.update_state(cost)

    @tf.function
    def val_step(self, val_ds):
        for x_batch_train, y_batch_train in val_ds:
            x_enc = self.encoder(x_batch_train, training=False)
            y_preds = self.task(x_enc, training=False)

            cost = self.loss(y_batch_train, y_preds)

            self.loss_metric.update_state(tf.reduce_mean(cost))
            self.metric.update_state(y_batch_train, y_preds)

    def fit(
        self,
        train_dataset,
        val_dataset,
        batch_size,
        num_epochs,
        model_directory,
        save_steps=10,
        function=np.min,
    ):
        """
        Fit deep learning methods using a custom training approach
        :param train_dataset (tuple of np.array): training data (x_train, y_train)
        :param val_dataset (tuple of np.array): training data (x_train, y_train)
        :param batch_size (int) : batch size for each epoch
        :param num_epochs (int) : number of epochs
        :param model_directory (str) : path to save the model weights and other metadata
        :param save_steps (int) : interval of epochs to save the "best" model
        :param function (np.function): function to optimize the loss (min or max)
        :return:
        """
        # Initialize loss to -inf if the purpose is to maximize, otherwise min
        global val_acc_result
        train_loss, val_loss, val_acc = (
            [np.inf] if function == np.min else [-np.inf] for i in range(3)
        )

        x_train, y_train = train_dataset
        x_val, y_val = val_dataset
        # Instance tf model (encoder, task, discriminator)
        _ = self(tf.zeros(list(x_val[:1, :, :].shape)))

        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

        for epoch in range(num_epochs + 1):
            # Shuffle at each epoch to avoid the model learning the order of the observations
            x_train_, y_train_ = shuffle(x_train, y_train)

            train_ds = tf.data.Dataset.from_tensor_slices((x_train_, y_train_)).batch(
                batch_size
            )

            self.train_step(train_ds)
            loss_epoch = self.loss_metric.result().numpy()
            train_loss.append(loss_epoch)
            self.loss_metric.reset_states()

            if epoch % save_steps == 0:
                self.val_step(val_ds)
                val_loss_epoch = self.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                print(
                    "Epoch {0}: Train loss {1}, Val loss {2}, Val acc {3}".format(
                        str(epoch),
                        str(loss_epoch),
                        str(round(val_loss_epoch, 4)),
                        str(round(val_acc_result, 4)),
                    )
                )

                if (
                    function is np.min
                    and val_loss_epoch < function(val_loss)
                    or function is np.max
                    and val_loss_epoch > function(val_loss)
                ):
                    print(f"Best score seen so far {str(val_loss_epoch)}")
                    self.encoder.save_weights(
                        os.path.join(model_directory, "encoder_best_model")
                    )
                    self.task.save_weights(
                        os.path.join(model_directory, "task_best_model")
                    )

                val_loss.append(val_loss_epoch)
                val_acc.append(val_acc_result)

        self.encoder.save_weights(os.path.join(model_directory, "encoder_last_model"))
        self.task.save_weights(os.path.join(model_directory, "task_last_model"))

        # History of the training
        losses = dict(train_loss_results=train_loss, val_loss_results=val_acc)
        with open(os.path.join(model_directory, "history.pickle"), "wb") as d:
            pickle.dump(losses, d, protocol=pickle.HIGHEST_PROTOCOL)

        # Save the config of the model
        config_model = dict(self.config)

        with open(os.path.join(model_directory, "config_model.pickle"), "wb") as d:
            pickle.dump(config_model, d, protocol=pickle.HIGHEST_PROTOCOL)
