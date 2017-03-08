# -*- coding: utf-8 -*-
import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sklearn.metrics

log = logging.getLogger(name=__name__)


def auc(truth, preds):
    """
    Calcuates area under the receiver operating characteristic curve

    Parameters
    ----------
    truth : iterable of bool
        True == non-native
    preds : iterable of float
        Probability that each instance is non-native

    Returns
    -------
    float
        AUC
    """
    return sklearn.metrics.roc_auc_score(truth, preds)


def pearsonr(truth, preds):
    """
    Calcuates the correlation

    Parameters
    ----------
    truth : iterable of bool
        True == non-native
    preds : iterable of float
        Probability that each instance is non-native

    Returns
    -------
    float
        Pearson's R
    """
    return scipy.stats.pearsonr(truth, preds)[0]


def calculate(df, preds):
    """
    Calcuates a dictionary of metrics

    Parameters
    ----------
    df : DataFrame
        True data
    preds : iterable of float
        Probability that each instance is non-native

    Returns
    -------
    dict
        Calculated matrics
    """
    ret = {}
    ret['AUC'] = auc(df.non_native.values, preds)
    ret['length_corr'] = pearsonr(preds, df.text.str.len().values)

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
    print('Length correlation: {}'.format(m['length_corr']))


def plot(truth, preds, model_type, path):
    """
    Visualize the ROC curve

    Parameters
    ----------
    truth : iterable of bool
        True == non-native
    preds : iterable of float
        Probability that each instance is non-native
    model_type : str
        The type of model that was used
    path : str
        Where to save the plot
    """
    truth = np.asarray(truth, dtype=np.float)

    # thresholds = np.linspace(0, 1.0, 100)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(truth, preds)
    auc = sklearn.metrics.auc(fpr, tpr)
    # log.debug(truth)
    # log.debug(preds)
    # log.debug(fpr)
    # log.debug(tpr)
    # log.debug(thresholds)

    fig = plt.figure()
    plt.clf()
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of {0}: AUC={1:0.2f}'.format(model_type, auc))
    plt.legend(loc="lower left")
    fig.savefig(path)
    log.info('ROC Curve saved. Open with: eog {}'.format(path))
