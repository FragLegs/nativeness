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
