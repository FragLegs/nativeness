# -*- coding: utf-8 -*-
import logging

import tensorflow as tf

import nativeness.models.pool

log = logging.getLogger(name=__name__)


class PromptAwareAvg(nativeness.models.pool.BiLSTMPoolAvg):
    """
    Extends the BiLSTMPool model to add extra loss related to the prompt
    that the essay was written to. This will hopefully remove some of the
    confound that is caused by differing words/topics across prompts.
    """
    def add_placeholders(self):
        super(PromptAwareAvg, self).add_placeholders()

        self.prompt_placeholder = tf.placeholder(
            tf.int32,
            shape=(None, ),  # n_windows
            name='prompt_label'
        )

    def create_feed_dict(self,
                         inputs_batch,
                         labels_batch=None,
                         keep_prob=1.0,
                         learning_rate=None,
                         dev_auc=0.5,
                         prompt_batch=None):
        feed_dict = super(PromptAwareAvg, self).create_feed_dict(
            inputs_batch, labels_batch, keep_prob, learning_rate, dev_auc
        )

        if prompt_batch is not None:
            feed_dict[self.prompt_placeholder] = prompt_batch

        return feed_dict

    def add_prompt_prediction_op(self):
        """
        Starting with the dropped output layer (the output of the BiLSTM),
        generate a (n_prompts, ) size tensor that is meant to predict the
        prompt that the essay came from.

        Returns
        -------
        tensor (n_prompts ,)
            Predicted prompt values (to be passed to a softmax)
        """
        self.Wp = tf.Variable(
            initial_value=tf.contrib.layers.xavier_initializer()((
                self.config.hidden_size * 2, self.config.n_prompts
            )),
            name='Wp'
        )
        self.bp = tf.Variable(
            initial_value=tf.zeros(self.config.n_prompts,),
            name='bp'
        )

        return tf.matmul(self.dropped_output, self.Wp) + self.bp

    def add_loss_op(self, pred, prompt_pred):
        """
        Adds Ops for the loss function to the computational graph.

        Parameters
        ----------
        pred: A (1, 1) tensor
            The nativeness prediction
        prompt_pred: a (n_prompts, ) tensor

        Returns:
            loss: A 0-d tensor (scalar) output
        """

        # compute the log loss of the nativeness prediction
        self.pred_loss = tf.contrib.losses.log_loss(
            labels=self.labels_placeholder, predictions=pred
        )

        self.prompt_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=prompt_pred, labels=self.prompt_placeholder
            )
        )

        return (
            self.pred_loss + (self.config.scale_prompt_loss * self.prompt_loss)
        )

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
        tf.summary.scalar('loss_prompt', self.prompt_loss)
        tf.summary.scalar('loss_pred', self.pred_loss)

        self.summary = tf.summary.merge_all()

    def build(self):
        self.add_placeholders()
        self.embeddings = self.add_embeddings()
        self.window_preds = self.add_window_prediction_op(self.embeddings)
        self.pred = self.add_prediction_op(self.window_preds)
        self.prompt_pred = self.add_prompt_prediction_op()
        self.loss = self.add_loss_op(self.pred, self.prompt_pred)
        self.train_op = self.add_training_op(self.loss)
        self.add_summaries()

    def w2d(self, window_preds):
        """
        Combine window predictions into a document prediction
        by taking the average

        Parameters
        ----------
        window_preds : iterable of float
            The predictions for the individual windows

        Returns
        -------
        float
            The document prediction
        """
        return window_preds.mean()


class PromptAwareMax(PromptAwareAvg):
    def w2d(self, window_preds):
        """
        Combine window predictions into a document prediction
        by taking the average

        Parameters
        ----------
        window_preds : iterable of float
            The predictions for the individual windows

        Returns
        -------
        float
            The document prediction
        """
        return window_preds.max()
