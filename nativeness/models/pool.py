# -*- coding: utf-8 -*-
import logging

import numpy as np
import tensorflow as tf

from nativeness.models.nn import NativeNN

log = logging.getLogger(name=__name__)


class BiLSTMPool(NativeNN):
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
            tf.uint8,
            shape=(
                None,  # batch size
                self.config.window_size,  # 100
            ),
            name='x'
        )
        self.labels_placeholder = tf.placeholder(
            tf.bool,
            shape=(1, 1),  # we train one doc at a time, w/ a batch of windows
            name='y'
        )

    def create_feed_dict(self, inputs_batch, labels_batch=None):
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
        inputs_batch: numpy array
            A batch of input data.
        labels_batch: numpy array, optional
            A batch of label data.

        Returns
        -------
        feed_dict: dict
            The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {
            self.input_placeholder: inputs_batch
        }

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        return feed_dict

    def add_embedding(self):
        """
        Adds an embedding layer
        Returns
        -------
            embeddings: tf.Tensor of shape (None, window_size, embed_size)
        """
        vocab = tf.Variable(
            initial_value=tf.contrib.layers.xavier_initializer()((
                self.config.vocab_size, self.config.embed_size
            )),
            name='vocab'
        )

        # features should be (None, window_size, embed_size)
        return tf.nn.embedding_lookup(vocab, self.input_placeholder)

    def add_window_prediction_op(self, embeddings):
        """
        Implements the core of the model that transforms a batch of input
        data into predictions.

        Returns
        -------
        A 1-d tensor (batch_size, )
            The window predictions for this document
        """
        # add a forward lstm cell
        lstm_fw = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size)
        lstm_fw.output_size

        # add a backwards lstm cell
        lstm_bw = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size)

        # add a bidirectional LSTM
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            lstm_fw, lstm_bw, embeddings
        )

        # concat the outputs and take the final output
        final_output = tf.slice(
            tf.concat(outputs, 2),
            [0, self.config.window_size - 1, 0],
            [-1, 1, -1]
        )

        # add an affine layer
        W = tf.Variable(
            initial_value=tf.contrib.layers.xavier_initializer()((
                self.config.output_size, 1
            )),
            name='W'
        )
        b = tf.Variable(
            initial_value=tf.zeros(1,),
            name='b'
        )

        # predict on the output
        return tf.sigmoid(tf.matmul(final_output, W) + b)

    def add_prediction_op(self, window_preds):
        """
        Average the windows

        Parameters
        ----------
        window_preds : A 1-d tensor (batch_size, )
             The window predictions for this document

        Returns
        -------
        A 0-d tensor (scalar)
            The document prediction
        """
        # average the result of the batch
        return tf.reduce_mean(window_preds)

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A 0-d tensor (scalar)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        # compute the log loss
        return tf.losses.log_loss(self.labels_placeholder, pred)

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
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config.learning_rate
        )
        return optimizer.minimize(loss=loss)