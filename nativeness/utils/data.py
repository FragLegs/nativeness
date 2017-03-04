# -*- coding: utf-8 -*-
import logging
import json
import os

import dill as pickle
import numpy as np
import pandas as pd

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
        'csv': _load_csv
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


def to_windows(char_ints, size=100):
    """
    Takes a list of character ints and returns k windows of `size` ints

    Parameters
    ----------
    char_ints : iterable of int
        The ints from to_ints()

    Returns
    -------
    list of list of ints
        The windows
    """
    return [char_ints[i:i + size] for i in range(len(char_ints) - size + 1)]


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
        One of: pickle, json, csv

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
        'csv': _save_csv
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
