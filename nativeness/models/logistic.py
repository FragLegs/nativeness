# -*- coding: utf-8 -*-
import logging

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.linear_model
import sklearn.pipeline

from nativeness.models.base import NativenessModel
import nativeness.utils.metrics
import nativeness.utils.text

log = logging.getLogger(name=__name__)


class LogisticWindows(NativenessModel):
    def train(self, train_data, dev_data):
        if hasattr(self, 'model'):
            return

        # set up feature extractor
        self.char_grams = CountVectorizer(ngram_range=(4, 4), analyzer='char')

        # get features and lables for train set
        log.debug('Fitting training features')

        essays = train_data.text.values
        self.char_grams.fit(essays)

        train_y = train_data.non_native.values.astype(np.float)
        classes = np.unique(train_y)
        dev_y = dev_data.non_native.values.astype(np.float)

        log.debug('Extracting dev features')
        dev_X = [
            self.char_grams.transform(nativeness.utils.text.windows(essay))
            for essay in dev_data.text.values
        ]

        best_auc = -np.inf
        best_model = None

        # try 10 regularization levels
        for c in np.logspace(start=-6, stop=2, num=10):
            log.debug('Trying c = {}'.format(c))

            # fit a model on train
            log.debug('Fitting')
            model = sklearn.linear_model.SGDClassifier(loss='log')

            for epoch in range(self.config.n_epochs):
                log.debug('Epoch {}'.format(epoch))

                windows = []
                labels = []
                n_essays = 100
                for i in range(n_essays):
                    _windows, _labels = nativeness.utils.text.random_windows(
                        essays, train_y
                    )
                    windows.extend(_windows)
                    labels.extend(_labels)

                log.debug('collected {} windows from {} essays'.format(
                    len(windows), n_essays
                ))
                log.debug('Making features')
                X = self.char_grams.transform(windows)

                log.debug('Training SGD')
                model.partial_fit(X, labels, classes=classes)

                log.debug('Testing against dev')

                # predict on dev
                log.debug('Predicting')
                dev_preds = [
                    model.predict_proba(dev_windows)[:, 1].ravel().mean()
                    for dev_windows in dev_X
                ]

                # get the AUC
                auc = nativeness.utils.metrics.auc(dev_y, dev_preds)

                # if this is a new best, save it
                if auc > best_auc:
                    log.info('New best AUC found! {0:0.2f} (c = {1})'.format(
                        auc, c
                    ))
                    best_auc = auc
                    best_model = model
                else:
                    log.debug('Not the best. {0:0.2f} (c = {1})'.format(
                        auc, c
                    ))

        self.model = best_model

    def predict(self, windows):
        X = self.char_grams.transform(windows)
        return self.model.predict_proba(X)[:, 1].ravel().mean()


class Logistic(NativenessModel):
    def train(self, train_data, dev_data):
        # set up feature extractor
        self.char_grams = CountVectorizer(ngram_range=(4, 4), analyzer='char')

        # get features and labls for train and dev set
        log.debug('Extracting training features')
        train_X = self.char_grams.fit_transform(train_data.text.values)
        train_y = train_data.non_native.values.astype(np.float)

        dev_y = dev_data.non_native.values.astype(np.float)

        n_docs, n_features = train_X.shape
        log.debug(
            'Found {} features in {} documents'.format(n_features, n_docs)
        )

        best_auc = -np.inf
        best_model = None

        # try 10 regularization levels
        for c in np.logspace(start=-6, stop=2, num=10):
            log.debug('Trying c = {}'.format(c))

            # fit a model on train
            log.debug('Fitting')
            model = sklearn.linear_model.LogisticRegression()
            model.fit(train_X, train_y)

            # predict on dev
            log.debug('Predicting')
            dev_preds = [
                model.predict_proba(
                    self.char_grams.transform(
                        nativeness.utils.text.windows(essay)
                    )
                )[:, 1].ravel().mean()
                for essay in dev_data.text.values
            ]

            # get the AUC
            auc = nativeness.utils.metrics.auc(dev_y, dev_preds)

            # if this is a new best, save it
            if auc > best_auc:
                log.info('New best AUC found! {0:0.2f} (c = {1})'.format(
                    auc, c
                ))
                best_auc = auc
                best_model = model
            else:
                log.debug('Not the best. {0:0.2f} (c = {1})'.format(
                    auc, c
                ))

        self.model = best_model

    def predict(self, windows):
        X = self.char_grams.transform(windows)
        return self.model.predict_proba(X)[:, 1].ravel().mean()
