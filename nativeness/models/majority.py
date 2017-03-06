# -*- coding: utf-8 -*-
import logging

from nativeness.models.base import NativenessModel

log = logging.getLogger(name=__name__)


class Majority(NativenessModel):
    def train(self, train_data, dev_data):
        self.majority = float(train_data.non_native.sum() >= len(train_data))

    def predict(self, windows):
        return self.majority
