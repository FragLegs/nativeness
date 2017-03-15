# -*- coding: utf-8 -*-
import logging

import numpy as np
import tensorflow as tf

from nativeness.models.nn import NativeNN

log = logging.getLogger(name=__name__)


class LogisticNN(NativeNN):
    """
    Represents each essay as 100 character sliding windows of text. Trains
    an BiLSTM for the windows and then averages the predictions with a pooling
    layer.
    """
    def add_placeholders(self):
        """
        Adds placeholder variables to tensorflow computational graph.

        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are
        used as inputs by the rest of the model building and will be fed data
        during training.

        See for more information:
        tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        self.input_placeholder = tf.placeholder(
            tf.float32,
            shape=(
                None,  # batch size
                self.config.ngram_params,  # 2**15
            ),
            name='x'
        )
        self.labels_placeholder = tf.placeholder(
            tf.bool,
            shape=(1, 1),  # we train one doc at a time, w/ a batch of windows
            name='y'
        )
        self.learning_rate_placeholder = tf.placeholder(
            tf.float32,
            shape=(),
            name='learning_rate'
        )
        self.dev_auc_placeholder = tf.placeholder(
            tf.float32,
            shape=(),
            name='dev_auc'
        )

    def create_feed_dict(self,
                         inputs_batch,
                         labels_batch=None,
                         keep_prob=1.0,
                         learning_rate=None,
                         dev_auc=0.5,
                         prompt_batch=None):
        """
        Creates the feed_dict for one step of training.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If labels_batch is None, then no labels are added to feed_dict.

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Parameters
        ----------
        inputs_batch: scipy sparse array
            A batch of input data.
        labels_batch: numpy array, optional
            A batch of label data.

        Returns
        -------
        feed_dict: dict
            The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {
            self.input_placeholder: inputs_batch,
            self.dev_auc_placeholder: dev_auc
        }

        if learning_rate is not None:
            feed_dict[self.learning_rate_placeholder] = learning_rate

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        return feed_dict

    def add_embeddings(self):
        """
        Adds an embedding layer
        Returns
        -------
            embeddings: tf.Tensor of shape (None, window_size, embed_size)
        """
        return None

    def add_window_prediction_op(self, embeddings):
        """
        Implements the core of the model that transforms a batch of input
        data into predictions.

        Returns
        -------
        A 1-d tensor (batch_size, )
            The window predictions for this document
        """
        # add an affine layer
        self.W = tf.Variable(
            initial_value=tf.contrib.layers.xavier_initializer()((
                self.config.ngram_params, 1
            )),
            name='W'
        )
        self.b = tf.Variable(
            initial_value=tf.zeros(1,),
            name='b'
        )

        # predict on the output
        return tf.sigmoid(tf.matmul(self.input_placeholder, self.W) + self.b)

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A 0-d tensor (scalar)
        Returns:
            loss: A 0-d tensor (scalar) output
        """

        # compute the log loss
        loss = tf.contrib.losses.log_loss(
            labels=self.labels_placeholder, predictions=pred
        ) + (self.config.l2_lambda * tf.nn.l2_loss(self.W))

        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable
        variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model. See

        tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Args:
            loss: Loss tensor (a scalar).
        Returns:
            train_op: The Op for training.
        """
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate_placeholder
        )
        return optimizer.minimize(loss=loss, global_step=self.global_step)


class LogisticAvg(LogisticNN):
    def add_prediction_op(self, window_preds):
        """
        Turn window preds into document preds by averaging

        Parameters
        ----------
        window_preds : A 1-d tensor (batch_size, )
             The window predictions for this document

        Returns
        -------
        A 0-d tensor (scalar)
            The document prediction
        """
        return tf.reshape(tf.reduce_mean(window_preds), (1, 1))


class LogisticMax(LogisticNN):
    def add_prediction_op(self, window_preds):
        """
        Turn window preds into document preds by averaging

        Parameters
        ----------
        window_preds : A 1-d tensor (batch_size, )
             The window predictions for this document

        Returns
        -------
        A 0-d tensor (scalar)
            The document prediction
        """
        return tf.reshape(tf.reduce_max(window_preds), (1, 1))
