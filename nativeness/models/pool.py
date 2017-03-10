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
            tf.int32,
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
        self.sequence_length = tf.placeholder(
            tf.int32,
            shape=(None, ),  # one length per window
            name='seq_len'
        )
        self.keep_prob_placeholder = tf.placeholder(
            tf.float32,
            shape=(),
            name='keep_prob'
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
            self.input_placeholder: inputs_batch,
            self.sequence_length: np.repeat(
                [self.config.window_size],
                inputs_batch.shape[0]
            ),
            self.keep_prob_placeholder: keep_prob,
            self.learning_rate_placeholder: learning_rate,
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
        self.vocab = tf.Variable(
            initial_value=tf.contrib.layers.xavier_initializer()((
                self.config.vocab_size, self.config.embed_size
            )),
            name='vocab'
        )

        # features should be (None, window_size, embed_size)
        embeddings = tf.nn.embedding_lookup(self.vocab, self.input_placeholder)

        # add dropout
        return tf.nn.dropout(embeddings, self.keep_prob_placeholder)

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

        # add a backwards lstm cell
        lstm_bw = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size)

        # add a bidirectional LSTM
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_fw,
            cell_bw=lstm_bw,
            inputs=embeddings,
            dtype=tf.float32,
            sequence_length=self.sequence_length
        )

        state_fw, state_bw = states
        final_c_fw, final_h_fw = state_fw
        final_c_bw, final_h_bw = state_bw
        final_output = tf.concat(1, (final_h_fw, final_h_bw))

        # # concat the outputs and take the final output
        # final_output = tf.reshape(
        #     tf.slice(
        #         tf.concat(2, outputs),
        #         [0, self.config.window_size - 1, 0],
        #         [-1, 1, -1]
        #     ),
        #     shape=(-1, self.config.rnn_output_size * 2)
        # )

        self.dropped_output = tf.nn.dropout(
            final_output, self.keep_prob_placeholder
        )

        # add an affine layer
        self.W = tf.Variable(
            initial_value=tf.contrib.layers.xavier_initializer()((
                self.config.hidden_size * 2, 1
            )),
            name='W'
        )
        self.b = tf.Variable(
            initial_value=tf.zeros(1,),
            name='b'
        )

        # predict on the output
        return tf.sigmoid(tf.matmul(self.dropped_output, self.W) + self.b)

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
        )

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


class BiLSTMPoolAvg(BiLSTMPool):
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


class BiLSTMPoolMax(BiLSTMPool):
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
