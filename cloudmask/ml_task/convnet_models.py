from marshmallow import fields
from marshmallow.validate import OneOf

from tensorflow.keras.layers import Dense
from cloudmask.base_models.base_tempnets import BaseCustomTempnetsModel
import tensorflow as tf


class CNN1D(BaseCustomTempnetsModel):
    """
    Implementation of the TempCNN network taken from the temporalCNN implementation
    https://github.com/charlotte-pel/temporalCNN
    """

    class CNN1DSchema(BaseCustomTempnetsModel._Schema):
        keep_prob = fields.Float(
            required=True,
            description="Keep probability used in dropout tf.keras.layers.",
            example=0.5,
        )
        keep_prob_conv = fields.Float(
            missing=0.8, description="Keep probability used in dropout tf.keras.layers."
        )
        kernel_size = fields.Int(
            missing=5, description="Size of the convolution kernels."
        )
        nb_conv_filters = fields.Int(
            missing=16, description="Number of convolutional filters."
        )
        nb_conv_stacks = fields.Int(
            missing=3, description="Number of convolutional blocks."
        )
        n_strides = fields.Int(missing=1, description="Value of convolutional strides.")

        nb_fc_neurons = fields.Int(
            missing=256, description="Number of Fully Connect neurons."
        )
        nb_fc_stacks = fields.Int(
            missing=2, description="Number of fully connected tf.keras.layers."
        )
        fc_activation = fields.Str(
            missing="relu",
            description="Activation function used in final FC tf.keras.layers.",
        )

        emb_layer = fields.String(
            missing="GlobalAveragePooling1D",
            validate=OneOf(["Flatten", "GlobalAveragePooling1D", "GlobalMaxPooling1D"]),
            description="Final layer after the convolutions.",
        )
        padding = fields.String(
            missing="SAME",
            validate=OneOf(["SAME", "VALID", "CAUSAL"]),
            description="Padding type used in convolutions.",
        )
        activation = fields.Str(
            missing="relu", description="Activation function used in final filters."
        )

        n_classes = fields.Int(missing=1, description="Number of classes")

        output_activation = fields.String(
            missing="linear", description="Output activation"
        )

        kernel_initializer = fields.Str(
            missing="he_normal", description="Method to initialise kernel parameters."
        )
        kernel_regularizer = fields.Float(
            missing=0.0, description="L2 regularization parameter."
        )
        enumerate = fields.Bool(
            missing=False, description="Increase number of filters across convolution"
        )
        str_inc = fields.Bool(missing=False, description="Increase strides")

        fc_dec = fields.Bool(missing=False, description="Decrease dense neurons")

        batch_norm = fields.Bool(
            missing=True, description="Whether to use batch normalisation."
        )
        factor = fields.Float(
            missing=1.0, description="Factor to multiply lambda for DANN."
        )
        adaptative = fields.Bool(missing=True, description="Adaptative lambda for DANN")

        n_disc_class = fields.Int(
            missing=2, description="Number of discriminator classes"
        )

    def _cnn_layer(self, net, i=0, first=False):

        dropout_rate = 1 - self.config.keep_prob_conv
        filters = self.config.nb_conv_filters
        kernel_size = self.config.kernel_size
        n_strides = self.config.n_strides

        if self.config.enumerate:
            filters = filters * (2**i)

        if self.config.str_inc:
            n_strides = 1 if first or kernel_size == 1 else 2

        layer = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=n_strides,
            padding=self.config.padding,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer),
        )(net)
        if self.config.batch_norm:
            layer = tf.keras.layers.BatchNormalization(axis=-1)(layer)

        layer = tf.keras.layers.Dropout(dropout_rate)(layer)
        layer = tf.keras.layers.Activation(self.config.activation)(layer)

        return layer

    def _embeddings(self, net):

        name = "embedding"
        if self.config.emb_layer == "Flatten":
            net = tf.keras.layers.Flatten(name=name)(net)
        elif self.config.emb_layer == "GlobalAveragePooling1D":
            net = tf.keras.layers.GlobalAveragePooling1D(name=name)(net)
        elif self.config.emb_layer == "GlobalMaxPooling1D":
            net = tf.keras.layers.GlobalMaxPooling1D(name=name)(net)
        return net

    def _fcn_layer(self, net, i=0):
        dropout_rate = 1 - self.config.keep_prob
        nb_neurons = self.config.nb_fc_neurons
        if self.config.fc_dec:
            nb_neurons /= 2**i
        layer_fcn = Dense(
            units=nb_neurons,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer),
        )(net)
        if self.config.batch_norm:
            layer_fcn = tf.keras.layers.BatchNormalization(axis=-1)(layer_fcn)

        layer_fcn = tf.keras.layers.Dropout(dropout_rate)(layer_fcn)
        if self.config.fc_activation:
            layer_fcn = tf.keras.layers.Activation(self.config.fc_activation)(layer_fcn)

        return layer_fcn

    def build(self, inputs_shape):
        """Build TCN architecture

        The `inputs_shape` argument is a `(N, T, D)` tuple where `N` denotes the number of samples, `T` the number of
        time-frames, and `D` the number of channels
        """
        x = tf.keras.layers.Input(inputs_shape[1:])

        net = x
        conv = self._cnn_layer(net, 0, first=True)
        for i, _ in enumerate(range(self.config.nb_conv_stacks - 1)):
            conv = self._cnn_layer(conv, i + 1)

        embedding = self._embeddings(conv)
        ################################################################

        fc_task = self._fcn_layer(embedding)
        # Task
        for i in range(1, self.config.nb_fc_stacks):
            fc_task = self._fcn_layer(fc_task, i)

        output = Dense(
            units=self.config.n_classes,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer),
        )(fc_task)
        output = tf.keras.layers.Softmax()(output)

        #############################

        fc_disc = self._fcn_layer(embedding)
        for i in range(1, self.config.nb_fc_stacks):
            fc_disc = self._fcn_layer(fc_disc, i)

        discriminator = Dense(
            units=self.config.n_disc_class,
            activation="softmax",
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer),
        )(fc_disc)
        discriminator = tf.keras.layers.Softmax()(discriminator)

        self.encoder = tf.keras.Model(inputs=x, outputs=embedding)
        self.task = tf.keras.Model(inputs=embedding, outputs=output)
        self.discriminator = tf.keras.Model(inputs=embedding, outputs=discriminator)

    def call(self, inputs, training=None):
        # Encoder, task and discriminator => useful for domain adaptation (library adapt)
        enc = self.encoder(inputs, training)
        task = self.task(enc, training)
        disc = self.discriminator(enc, training)
        return enc, task, disc
