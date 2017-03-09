# -*- coding: utf-8 -*-
import logging
import json
import os

import dill as pickle
import numpy as np
import pandas as pd
import sklearn.utils

log = logging.getLogger(name=__name__)


def load(path, as_type='csv'):
    """
    Loads the data

    Parameters
    ----------
    path : str
        Location of data on disk
    as_type : str
        One of: pickle, json, csv

    Returns
    -------
    loaded object
    """
    # load the particular type of object
    loaders = {
        'pickle': _load_pickle,
        'json': _load_json,
        'csv': _load_csv,
    }
    return loaders[as_type](path)


def _load_csv(path):
    """
    Loads the data

    Parameters
    ----------
    path : str
        Location of data on disk

    Returns
    -------
    DataFrame
    """
    return pd.read_csv(path, encoding='utf8', index_col=0)


def _load_pickle(path):
    """
    Loads the data

    Parameters
    ----------
    path : str
        Location of data on disk

    Returns
    -------
    obj
    """
    with open(path, 'r') as fin:
        return pickle.load(fin)


def _load_json(path):
    """
    Loads the data

    Parameters
    ----------
    path : str
        Location of data on disk

    Returns
    -------
    obj
    """
    with open(path, 'r') as fin:
        return json.load(fin)


def to_ints(essay):
    """
    Turns a string into a list of numbers between 0 and 94

    Parameters
    ----------
    essay : str
        The text of an essay

    Returns
    -------
    numpy array of int
        The decimal version of each character, minus Space (\x32)
    """
    return np.array(map(ord, essay)) - 32


def to_char(char_int):
    """
    Turns a int into its character

    Parameters
    ----------
    char_int : int
        The to_ints modified character

    Returns
    -------
    char
        The character
    """
    return chr(char_int + 32)


def to_inputs(row):
    """
    Returns the inputs to the network

    Parameters
    ----------
    row : Series
        The row of essay data

    Returns
    ------
    tuple
        (text window (list of int), non-native (bool), prompt(int))
    """
    return (
        to_ints(row.text),
        row.non_native,
        row.prompt_index if 'prompt_index' in row.index else None
    )


def debug(df):
    """
    Returns a reduced version of the data frame

    Parameters
    ----------
    df : DataFrame
        One of the data sets

    Returns
    -------
    DataFrame
        10 instances of data
    """
    return pd.concat([df[df.non_native].iloc[:5], df[~df.non_native].iloc[:5]])


class WindowGenerator(object):
    def __init__(self, path, window_size, window_stride, debug, shuffle=False):
        """
        Parameters
        ----------
        path : str
            Where the data set is located
        window_size : int
            How big the windows should be
        window_stride : int
            How many steps between windows
        debug : bool
            Whether to limit the training/dev data
        shuffle : bool, optional
            Whether to shuffle the data
        """
        self.path = path
        self.window_size = window_size
        self.window_stride = window_stride
        self.debug = debug
        self.shuffle = shuffle

    def __call__(self, no_windows=False, no_ints=False):
        """
        Yields essays represented as windows of character ints with labels

        Parameters
        ----------
        no_windows : bool, optional
            If True, don't make the windows

        no_ints : bool, optional
            If True, don't convert to ints (i.e. for Logistic)

        Yields
        -------
        tuple (iterable of iterable of int, bool, int)
            - The windows of text represented as iterables of ints
            - Whether this is an ELL writer (True) or not
            - The prompt index (in case we need to control for it)
        """
        # load the data
        log.debug('Loading data from {}'.format(self.path))
        df = load(self.path, as_type='csv')

        has_prompt = 'prompt_index' in df.columns

        # only use 10 instances if we're debugging
        if self.debug:
            df = debug(df)

        # shuffle the data, if desired
        if self.shuffle:
            df = sklearn.utils.shuffle(df)

        for index, row in df.iterrows():
            windows = None if no_windows else (
                to_windows(row.text) if no_ints else
                to_windows(to_ints(row.text))
            )
            label = row.non_native
            prompt = row.prompt_index if has_prompt else None

            yield windows, label, prompt

    def size(self):
        """
        Report the size of the data set

        Returns
        -------
        int
            how many instances there are
        """
        try:
            return self._size
        except AttributeError:
            # load the data
            log.debug('Loading data from {}'.format(self.path))
            df = load(self.path, as_type='csv')

            # only use 10 instances if we're debugging
            if self.debug:
                df = debug(df)

            self._size = len(df)

            return self._size


def train_generator(config):
    path = os.path.join(config.input_path, 'train.csv')
    return WindowGenerator(
        path,
        config.window_size,
        config.window_stride,
        config.debug,
        shuffle=True
    )


def dev_generator(config):
    path = os.path.join(config.input_path, 'dev.csv')
    return WindowGenerator(
        path,
        config.window_size,
        config.window_stride,
        config.debug,
    )


def test_generator(config):
    path = os.path.join(config.input_path, 'test.csv')
    return WindowGenerator(
        path,
        config.window_size,
        config.window_stride,
        config.debug,
    )


def to_windows(char_ints, size=100, stride=1):
    """
    Takes a list of character ints and returns k windows of `size` ints

    Parameters
    ----------
    char_ints : iterable of int
        The ints from to_ints()

    size : int
        Size of the window

    stride : int
        How many characters between each window

    Returns
    -------
    list of list of ints
        The windows
    """
    return [
        char_ints[i:i + size]
        for i in range(0, len(char_ints) - size + 1, stride)
    ]


def save(path, obj, name, as_type='pickle'):
    """
    Saves an object to disk

    Parameters
    ----------
    path : str
        The directory to save in
    obj : object
        The object to save
    name : str
        What to call it
    as_type : str
        One of: pickle, json, csv, npy

    Returns
    -------
    str
        Saved path
    """
    saved_path = os.path.join(path, name)

    # save the particular type of object
    savers = {
        'pickle': _save_pickle,
        'json': _save_json,
        'csv': _save_csv,
        'npy': _save_npy
    }
    savers[as_type](saved_path, obj)

    return saved_path


def _save_pickle(path, obj):
    with open(path, 'w') as fout:
        pickle.dump(obj, fout)


def _save_json(path, obj):
    with open(path, 'w') as fout:
        json.dump(obj, fout)


def _save_csv(path, obj):
    obj.to_csv(path, encoding='utf8', index=True)


def _save_npy(path, obj):
    with open(path, 'w') as fout:
        np.save(fout, obj)
