# -*- coding: utf-8 -*-
import logging
import operator
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from nativeness.models.base import NativenessModel
import nativeness.utils.data
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

    def create_feed_dict(self, inputs_batch, labels_batch=None, keep_prob=1.0):
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

    def train_on_batch(self,
                       sess,
                       inputs_batch,
                       labels_batch,
                       lr,
                       auc,
                       prompt_batch=None):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(
            inputs_batch,
            labels_batch=labels_batch,
            keep_prob=self.config.keep_prob,
            learning_rate=lr,
            dev_auc=auc,
            prompt_batch=prompt_batch
        )
        _, loss, _, summary, global_step = sess.run(
            [
                self.train_op,
                self.loss,
                self.update_train_auc,
                self.summary,
                self.global_step
            ],
            feed_dict=feed
        )
        return loss, summary, global_step

    def predict_on_batch(self, sess, inputs_batch, labels_batch=None):
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
        feed = self.create_feed_dict(
            inputs_batch, labels_batch=labels_batch, keep_prob=1.0
        )

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
        self.add_summaries()

        log.debug('{} parameters'.format(self.count_parameters()))

    def count_parameters(self):
        """
        Count the number of trainable parameters in the model

        Returns
        -------
        int
        """
        return sum([
            reduce(operator.mul, layer.get_shape().as_list())
            for layer in tf.trainable_variables()
        ])

    def add_summaries(self):

        # update the auc
        self.train_auc, self.update_train_auc = (
            tf.contrib.metrics.streaming_auc(
                predictions=self.pred,
                labels=self.labels_placeholder,
                curve='ROC',
                name='train_auc'
            )
        )

        tf.summary.scalar('auc_train', self.train_auc)
        tf.summary.scalar('auc_dev', self.dev_auc_placeholder)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar(
            'learning_rate',
            self.learning_rate_placeholder
        )
        tf.summary.scalar('pred', tf.reduce_mean(self.pred))
        tf.summary.scalar(
            'off_by',
            tf.reduce_mean(
                tf.abs(
                    tf.cast(self.labels_placeholder, tf.float32) - self.pred
                )
            )
        )

        self.summary = tf.summary.merge_all()

    def run_epoch(self, sess, train_generator, dev_generator, lr, writer, auc):
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
            loss, summary, global_step = self.train_on_batch(
                sess=sess,
                inputs_batch=np.array(windows),
                labels_batch=[[label]],
                lr=lr,
                auc=auc,
                prompt_batch=np.repeat([prompt_id], len(windows))
            )
            prog.update(i, [('loss', loss)])
            writer.add_summary(summary, global_step)
            if i >= self.config.max_essays_per_epoch:
                break

        log.debug('Predicting on dev')

        # set up the progress bar
        prog = nativeness.utils.progress.Progbar(target=dev_generator.size())
        i = 0

        # reset dev AUC
        # dev_scope = tf.get_collection(
        #     tf.GraphKeys.TRAINABLE_VARIABLES, 'dev'
        # )
        # sess.run(tf.variables_initializer(dev_scope))

        # train
        preds = []
        truth = []
        for windows, label, prompt_id in dev_generator():
            i += 1
            truth.append(label)
            pred, _ = self.predict_on_batch(
                sess, np.array(windows), [[label]]
            )
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

        with tf.Graph().as_default() as graph:
            # build the model
            log.debug('Building the model')
            tf.set_random_seed(self.config.random_seed)
            self.build()

            # initialize variables
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

            log_outputs = os.path.join(self.config.results_path, 'logs')
            writer = tf.summary.FileWriter(log_outputs, graph=graph)

            # add the vocab to the summary writer
            if self.embeddings is not None:
                vocab_path = os.path.join(log_outputs, 'vocabulary.tsv')
                df = pd.DataFrame()
                df['char'] = [
                    nativeness.utils.data.to_char(i)
                    for i in range(self.config.vocab_size)
                ]
                df.char = df.char.str.replace(' ', 'SP')  # easier to see
                df['embedding_index'] = range(len(df))
                df.to_csv(vocab_path, sep='\t', index=False)
                projector_config = projector.ProjectorConfig()
                embedding = projector_config.embeddings.add()
                embedding.tensor_name = self.vocab.name
                embedding.metadata_path = vocab_path
                projector.visualize_embeddings(writer, projector_config)

            # set up a session
            session_config = (
                tf.ConfigProto(log_device_placement=True)
                if self.config.log_device else None
            )
            with tf.Session(config=session_config) as session:
                session.run(init)

                # reset auc
                session.run(tf.local_variables_initializer())

                learning_rate = self.config.lr
                auc = 0.5

                for e in range(self.config.n_epochs):
                    log.info('Epoch {}'.format(e))
                    auc, preds = self.run_epoch(
                        session,
                        train_generator,
                        dev_generator,
                        learning_rate,
                        writer,
                        auc
                    )

                    learning_rate *= self.config.lr_delta

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

            # simple test to see if we're running on GPUs
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
