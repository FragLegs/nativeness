# -*- coding: utf-8 -*-
import logging

import numpy as np

log = logging.getLogger(name=__name__)


def windows(essay, size=100):
    return [essay[i:i + size] for i in range(len(essay) - size + 1)]


def random_windows(essays, labels, size=100):
    i = np.random.randint(len(essays))
    w = windows(essays[i], size=size)
    return w, [labels[i]] * len(w)
    # for essay, label in zip(essays, labels):
    #     windows = [essay[i:i + size] for i in range(len(essay) - size + 1)]
    #     window_labels = [label] * len(windows)
    #     yield windows, window_labels
