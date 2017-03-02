# -*- coding: utf-8 -*-
import logging

import sklearn.metrics

log = logging.getLogger(name=__name__)


def calculate(truth, preds):
    """
    Calcuates a dictionary of metrics

    Parameters
    ----------
    truth : iterable of bool
        True == non-native
    preds : iterable of float
        Probability that each instance is non-native

    Returns
    -------
    dict
        Calculated matrics
    """
    ret = {}
    ret['AUC'] = sklearn.metrics.roc_auc_score(truth, preds)

    return ret


def show(m):
    """
    Pretty prints the metrics

    Parameters
    ----------
    m : dict
        Calculated metrics
    """
    print('AUC: {}'.format(m['AUC']))
