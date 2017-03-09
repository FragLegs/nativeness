# -*- coding: utf-8 -*-
import logging
import os

import numpy as np

from nativeness.models.base import NativenessModel
import nativeness.utils.data


log = logging.getLogger(name=__name__)


class Majority(NativenessModel):
    def train(self, train_generator, dev_generator):
        _, labels, _ = zip(*train_generator(no_windows=True, no_ints=True))
        self.majority = float(sum(labels) >= len(labels))

        self.save_model()

        _, labels, _ = zip(*dev_generator(no_windows=True, no_ints=True))
        return np.repeat([self.majority], len(labels))

    def predict(self, test_generator):
        preds = []
        window_preds = []

        for windows, _, _ in test_generator(no_ints=True):
            preds.append(self.majority)
            window_preds.append(np.repeat([self.majority], len(windows)))

        return np.array(preds), window_preds

    def save_model(self):
        model = {
            'majority': self.majority,
        }
        nativeness.utils.data.save(
            self.config.results_path,
            model,
            'model',
            as_type='pickle'
        )

    def load(self, weights_path):
        model = nativeness.utils.data.load(
            os.path.join(self.config.results_path, 'model'),
            as_type='pickle'
        )
        self.majority = model['majority']
