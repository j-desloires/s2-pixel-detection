import os

import tensorflow as tf

from cloudmask.base_models.base_custom_training import (
    BaseModelCustomTraining,
)


from marshmallow import Schema, fields
from marshmallow.validate import OneOf, ContainsOnly

from eoflow.models.losses import CategoricalCrossEntropy, CategoricalFocalLoss
from eoflow.models.metrics import InitializableMetric


# Available losses. Add keys with new losses here.
dictionary_losses = {
    "cross_entropy": CategoricalCrossEntropy,
    "focal_loss": CategoricalFocalLoss,
}

# Available metrics. Add keys with new metrics here.
dictionary_metrics = {
    "accuracy": tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
    "precision": tf.keras.metrics.Precision,
    "recall": tf.keras.metrics.Recall,
    "auc": tf.keras.metrics.AUC(multi_label=True),
}


class BaseCustomTempnetsModel(BaseModelCustomTraining):
    """Base for pixel-wise classification base_models."""

    class _Schema(Schema):
        # n_outputs = fields.Int(required=True, description='Number of output layers', example=1)
        learning_rate = fields.Float(
            missing=None, description="Learning rate used in training.", example=0.001
        )
        loss = fields.String(
            missing="cross_entropy",
            description="Loss function used for training.",
            validate=OneOf(dictionary_losses.keys()),
        )
        metrics = fields.String(
            missing="auc",
            description="List of metrics used for evaluation.",
            validate=OneOf(dictionary_metrics.keys()),
        )
        ema = fields.Bool(missing=True, description="Whether to use ema.")

    def prepare(
        self,
        optimizer=None,
        loss=None,
        metrics=None,
        loss_metric=tf.keras.metrics.Mean(),
        reduce_lr=False,
        **kwargs,
    ):
        """Prepares the model. Optimizer, loss and metrics are read using the following protocol:
        * If an argument is None, the default value is used from the configuration of the model.
        * If an argument is a key contained in segmentation specific losses/metrics, those are used.
        * Otherwise the argument is passed to `compile` as is.

        """
        # Read defaults if None
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config.learning_rate
            )

        if loss is None:
            loss = self.config.loss
        loss = dictionary_losses[loss](reduction=tf.keras.losses.Reduction.NONE)
        self.loss_metric = loss_metric

        if metrics is None:
            metrics = self.config.metrics
        self.metric = dictionary_metrics[metrics](**kwargs)

        if isinstance(self.metric, InitializableMetric):
            self.metric.init_from_config(self.config)

        self.compile(optimizer=optimizer, loss=loss, metrics=self.metric, **kwargs)

    # Override default method to add prediction visualization
    def train_and_evaluate(
        self,
        train_dataset,
        val_dataset,
        num_epochs,
        save_steps,
        model_directory,
        **kwargs,
    ):

        super().train_and_evaluate(
            train_dataset,
            val_dataset,
            num_epochs,
            save_steps,
            model_directory,
            **kwargs,
        )
