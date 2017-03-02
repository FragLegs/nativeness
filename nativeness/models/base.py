# -*- coding: utf-8 -*-
import abc
import logging
import os
from datetime import datetime


log = logging.getLogger(name=__name__)


class Config(object):
    def __init__(self, **kwargs):
        # add all keyword arguments to namespace
        self.__dict__.update(kwargs)

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

        log.debug('Recording results at {}'.format(self.results_path))


class NativenessModel(object):
    """
    Base class for all models
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def train(self, train_data, dev_data):
        """
        Calls the model's `_train()` class

        Parameters
        ----------
        train_data : DataFrame
            The training data

        dev_data : DataFrame
            The development data
        """

    @abc.abstractmethod
    def predict(self, essays):
        """
        Uses the model to predict on data

        Parameters
        ----------
        essays : iterable of str
            The essays to predict on

        Returns
        -------
        List of probabilities, one per row in the data set
        """
