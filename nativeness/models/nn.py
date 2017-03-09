# -*- coding: utf-8 -*-
import logging
import os

import numpy as np
import tensorflow as tf

from nativeness.models.base import NativenessModel
import nativeness.utils.metrics
import nativeness.utils.progress

log = logging.getLogger(name=__name__)


class NativeNN(NativenessModel):
    """
    A base model for Neural Networks implemented in Tensorflow. Uses the model
    structure provided in cs224n.
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
        raise NotImplementedError("Each Model must re-implement this method.")

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
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_window_prediction_op(self, embeddings):
        """
        Implements the core of the model that transforms a batch of input
        data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, )
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_prediction_op(self, window_preds):
        """
        Turn window preds into document preds

        Parameters
        ----------
        window_preds : A 1-d tensor (batch_size, )
             The window predictions for this document

        Returns
        -------
        A 0-d tensor (scalar)
            The document prediction
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")

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

        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, window_size)
        Returns
        -------
        tuple
            prediction: float
            window_preds : A 1-d array (batch_size, )
        """
        feed = self.create_feed_dict(inputs_batch)
        prediction, window_preds = sess.run(
            [self.pred, self.window_preds], feed_dict=feed
        )
        return prediction.ravel()[0], window_preds.ravel()

    def build(self):
        self.add_placeholders()
        self.embeddings = self.add_embeddings()
        self.window_preds = self.add_window_prediction_op(self.embeddings)
        self.pred = self.add_prediction_op(self.window_preds)
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def run_epoch(self, sess, train_generator, dev_generator):
        """
        Run a single epoch
        """
        # set up the progress bar
        prog = nativeness.utils.progress.Progbar(
            target=min(
                self.config.max_essays_per_epoch,
                train_generator.size()
            )
        )
        i = 0

        # train
        for windows, label, prompt_id in train_generator():
            i += 1
            loss = self.train_on_batch(sess, np.array(windows), [[label]])
            prog.update(i, [('loss', loss)])
            if i >= self.config.max_essays_per_epoch:
                break

        log.debug('Predicting on dev')

        # set up the progress bar
        prog = nativeness.utils.progress.Progbar(target=dev_generator.size())
        i = 0

        # train
        preds = []
        truth = []
        for windows, label, prompt_id in dev_generator():
            i += 1
            truth.append(label)
            pred, _ = self.predict_on_batch(sess, np.array(windows))
            preds.append(pred)
            prog.update(i)

        truth = np.array(truth)
        preds = np.array(preds)

        auc = nativeness.utils.metrics.auc(truth, preds)
        return auc, preds

    def train(self, train_generator, dev_generator):
        """
        Uses the neural net to learn to predict whether an essay was written
        by a non-native speaker.

        Train the logistic regression model with train and dev data

        Parameters
        ----------
        train_generator : nativeness.utils.data.WindowGenerator
            An object that can generate training windows
        dev_generator : nativeness.utils.data.WindowGenerator
            An object that can generate dev windows

        Returns
        -------
        iterable of float
            Dev predictions
        """
        log.debug(
            'Building model with {} training and {} dev instances'.format(
                train_generator.size(), dev_generator.size()
            )
        )

        best_auc = -np.inf
        best_preds = None

        with tf.Graph().as_default():
            # build the model
            log.debug('Building the model')
            self.build()

            # initialize variables
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

            # set up a session
            session_config = (
                tf.ConfigProto(log_device_placement=True)
                if self.config.log_device else None
            )
            with tf.Session(config=session_config) as session:
                session.run(init)

                for e in range(self.config.n_epochs):
                    log.info('Epoch {}'.format(e))
                    auc, preds = self.run_epoch(
                        session, train_generator, dev_generator
                    )

                    if auc > best_auc:
                        log.info('New best AUC found! {0:0.2f}'.format(auc))
                        best_auc = auc
                        best_preds = preds
                        model_path = os.path.join(
                            self.config.results_path, 'model.weights'
                        )
                        log.info('Saving model at {}'.format(model_path))
                        saver.save(session, model_path)
                    else:
                        log.debug('Not the best. {0:0.2f}'.format(auc))

        return best_preds

    def predict(self, test_generator):
        """
        Predict the probability that essays were written by ELL students

        Parameters
        ----------
        test_generator : nativeness.utils.data.WindowGenerator
            An object that can generate test window

        Returns
        -------
        tuple
            iterable of float : predicitons
            iterable of iterable of float : window predictions
        """
        with tf.Graph().as_default():
            self.build()
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            session_config = (
                tf.ConfigProto(log_device_placement=True)
                if self.config.log_device else None
            )
            with tf.Session(config=session_config) as session:
                session.run(init)
                model_path = os.path.join(
                    self.config.results_path, 'model.weights'
                )
                saver.restore(session, model_path)

                prog = nativeness.utils.progress.Progbar(
                    target=test_generator.size()
                )
                i = 0

                preds = []
                window_preds = []
                for windows, _, _ in test_generator():
                    i += 1
                    pred, wp = self.predict_on_batch(
                        session, np.array(windows)
                    )
                    preds.append(pred)
                    window_preds.append(wp)
                    prog.update(i)

                preds = np.array(preds)
                return preds, window_preds
