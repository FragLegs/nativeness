# -*- coding: utf-8 -*-
import logging

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
import sklearn.linear_model
import sklearn.pipeline

from nativeness.models.base import NativenessModel
import nativeness.utils.metrics
import nativeness.utils.progress
import nativeness.utils.text

log = logging.getLogger(name=__name__)


class Logistic(NativenessModel):
    # @profile
    def ngrams(self, window):
        n = self.config.ngram_size
        return [window[i:i + n] for i in range(len(window) - n + 1)]

    # @profile
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
        if hasattr(self, 'model'):
            return

        # set up feature extractor
        self.extractor = HashingVectorizer(
            analyzer=self.ngrams, n_features=2**15
        )

        dev = []
        dev_size = dev_generator.size()
        prog = nativeness.utils.progress.Progbar(target=dev_size)
        i = 0
        for windows, label, _ in dev_generator(no_ints=True):
            i += 1
            dev.append((self.extractor.transform(windows), label))
            prog.update(i)

        best_auc = -np.inf
        best_model = None
        best_preds = None

        # fit a model on train
        log.debug('Fitting')
        model = sklearn.linear_model.SGDClassifier(loss='log')

        essays_per_epoch = 1000

        for epoch in range(self.config.n_epochs):
            log.debug('Epoch {}'.format(epoch))

            log.debug('Training SGD')

            train_prog = nativeness.utils.progress.Progbar(
                target=essays_per_epoch
            )

            i = 0
            for windows, label, _ in train_generator(no_ints=True):
                i += 1
                X = self.extractor.transform(windows)
                y = np.repeat([label], X.shape[0])
                model.partial_fit(X, y, classes=[0, 1])
                train_prog.update(i)
                if i >= essays_per_epoch:
                    break

            log.debug('Testing against dev')

            dev_prog = nativeness.utils.progress.Progbar(target=len(dev))
            dev_preds = []
            dev_y = []
            i = 0
            for dev_X, label in dev:
                i += 1
                dev_y.append(label)
                preds = model.predict_proba(dev_X)
                dev_preds.append(self.w2d(preds[:, 1].ravel()))
                dev_prog.update(i)
            dev_preds = np.asarray(dev_preds)
            dev_y = np.asarray(dev_y)

            # get the AUC
            auc = nativeness.utils.metrics.auc(dev_y, dev_preds)

            # if this is a new best, save it
            if auc > best_auc:
                log.info('New best AUC found! {0:0.2f}'.format(auc))
                best_auc = auc
                best_model = model
                best_preds = dev_preds
            else:
                log.debug('Not the best. {0:0.2f}'.format(auc))

        self.model = best_model
        return best_preds

    def w2d(self, window_preds):
        """
        Combine window predictions into a document prediction by averaging them

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
        test_size = test_generator.size()
        test_prog = nativeness.utils.progress.Progbar(target=test_size)

        window_preds = []
        preds = []
        i = 0
        for windows, _, _ in test_generator(no_ints=True):
            i += 1
            X = self.extractor.transform(windows)
            window_preds.append(self.model.predict_proba(X)[:, 1].ravel())
            preds.append(self.w2d(window_preds[-1]))
            test_prog.update(i)

        preds = np.array(preds)

        return preds, window_preds
