# -*- coding: utf-8 -*-
from datetime import datetime
import logging
import os
import random

import numpy as np


log = logging.getLogger(name=__name__)


class Config(object):
    def __init__(self, **kwargs):
        self.n_epochs = 10
        self.learning_rate = 0.001
        self.embed_size = 32
        self.hidden_size = 64
        self.rnn_output_size = self.hidden_size
        self.vocab_size = 126 - 31  # the printable ascii characters

        self.window_size = 100
        self.window_stride = 1

        self.ngram_size = 4
        self.max_essays_per_epoch = 5000

        self.random_seed = 42

        self.log_device = False

        self.extra = []

        # add all keyword arguments to namespace
        self.__dict__.update(kwargs)
        for e in self.extra:
            if '=' in e:
                k, v = e.split('=')
                self.__dict__[k] = eval(v)
            else:
                self.__dict__[e] = True

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

        # set the random seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)


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
