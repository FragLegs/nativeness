# -*- coding: utf-8 -*-
from datetime import datetime
import logging
import os
import random

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


log = logging.getLogger(name=__name__)


class Config(object):
    def __init__(self, **kwargs):
        self.n_epochs = 12
        self.lr = 0.001
        self.lr_delta = 0.9
        self.embed_size = 64
        self.hidden_size = 128
        self.rnn_output_size = self.hidden_size
        self.vocab_size = 126 - 31  # the printable ascii characters
        self.keep_prob = 0.9

        self.window_size = 100
        self.window_stride = 5

        self.ngram_size = 4
        self.ngram_params = 2**15
        self.l2_lambda = 0.001

        self.max_essays_per_epoch = 20000

        self.random_seed = None

        self.log_device = False
        self.in_memory = True

        # this needs to be updated if the training data changes
        self.n_prompts = 95

        # weight for the weighted sum of losses
        self.scale_prompt_loss = 1.0

        # add all keyword arguments to namespace
        self.__dict__.update(kwargs)
        for e in self.extra:
            if '=' in e:
                k, v = e.split('=')
                self.__dict__[k] = eval(v)
            else:
                self.__dict__[e] = True

        if self.debug:
            self.max_essays_per_epoch = 10

        if self.random_seed is None:
            self.random_seed = random.randint(0, 10000)

        if self.reload is not None:
            self.results_path = self.reload
        else:
            # get a timestamp
            time = '{:%Y%m%d_%H%M%S}'.format(datetime.now())

            # results_path = output_path/model_type/time
            self.results_path = os.path.join(
                self.output_path, self.model_type, time
            )

        # make sure the path exists
        try:
            os.makedirs(self.results_path)
        except:
            pass

        root_log = logging.getLogger()
        log_path = os.path.join(self.results_path, 'log.txt')
        log_to_file = logging.FileHandler(log_path)
        root_log.addHandler(log_to_file)

        log.debug('Recording results at {}'.format(self.results_path))
        log.debug('Random seed is {}'.format(self.random_seed))

        # set the random seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.featurizer = None
        if 'logistic' in self.model_type:
            log.debug('Using Hashing Vectorizer')
            self.featurizer = HashingVectorizer(
                analyzer=self.ngrams, n_features=self.ngram_params
            )

    def ngrams(self, window):
        n = self.ngram_size
        return [window[i:i + n] for i in range(len(window) - n + 1)]


class NativenessModel(object):
    """
    Base class for all models
    """
    def __init__(self, config):
        self.config = config

    def train(self, train_generator, dev_generator):
        """
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
        raise NotImplementedError('Must implement train()')

    def predict(self, test_generator):
        """
        Predict the probability that essays were written by ELL students

        Parameters
        ----------
        test_generator : nativeness.utils.data.WindowGenerator
            An object that can generate test window

        Returns
        -------
        iterable of float
            Dev predictions
        """
        raise NotImplementedError('Must implement predict()')

    def load(weight_path):
        pass
