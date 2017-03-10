# -*- coding: utf-8 -*-
import logging

import tensorflow as tf

import nativeness.models.pool

log = logging.getLogger(name=__name__)


class PromptAwareBiLSTMPool(nativeness.models.pool.BiLSTMPool):
    """
    Extends the BiLSTMPool model to add extra loss related to the prompt
    that the essay was written to. This will hopefully remove some of the
    confound that is caused by differing words/topics across prompts.
    """
    def add_placeholders(self):
        super(PromptAwareBiLSTMPool, self).add_placeholders()

        self.prompt_placeholder = tf.placeholder(
            tf.int32,
            shape=(self.config.n_prompts, ),
            name='prompt_label'
        )

    def create_feed_dict(self,
                         inputs_batch,
                         labels_batch=None,
                         keep_prob=1.0,
                         learning_rate=None,
                         dev_auc=0.5,
                         prompt_batch=None):
        feed_dict = super(PromptAwareBiLSTMPool, self).create_feed_dict(
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

        # compute the log loss
        loss = tf.contrib.losses.log_loss(
            labels=self.labels_placeholder, predictions=pred
        )
