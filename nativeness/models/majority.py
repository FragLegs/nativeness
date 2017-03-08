# -*- coding: utf-8 -*-
import logging

import numpy as np

from nativeness.models.base import NativenessModel

log = logging.getLogger(name=__name__)


class Majority(NativenessModel):
    def train(self, train_generator, dev_generator):
        _, labels, _ = zip(*train_generator(no_windows=True))
        self.majority = float(sum(labels) >= len(labels))

        _, labels, _ = zip(*dev_generator(no_windows=True))
        return np.repeat([self.majority], len(labels))

    def predict(self, windows):
        return self.majority
