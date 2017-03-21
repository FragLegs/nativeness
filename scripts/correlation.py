# -*- coding: utf-8 -*-
import argparse
import copy
import logging
import os

import numpy as np
import pandas as pd

import nativeness.utils.data
from nativeness.utils.metrics import auc
import nativeness.utils.progress

log = logging.getLogger(name=__name__)


def main(**kwargs):
    df = pd.read_csv(
        '/research/ella/nativeness/final_results/scored.csv',
        encoding='utf8',
        index_col=0
    )

    ann = df.annotation.values

    print('Logistic Max: {}'.format(auc(ann, df.logistic_nn_max.values)))
    print('Logistic Avg: {}'.format(auc(ann, df.logistic_nn_avg.values)))
    print('Pool Max: {}'.format(auc(ann, df.pool_max.values)))
    print('Pool Avg: {}'.format(auc(ann, df.pool_avg.values)))
    print('Prompt Max: {}'.format(auc(ann, df.prompt_max.values)))
    print('Prompt Avg: {}'.format(auc(ann, df.prompt_avg.values)))




def parse_args():
    """
    Parses the arguments from the command line

    Returns
    -------
    argparse.Namespace
    """
    desc = 'Correlation between annotations and scores'
    parser = argparse.ArgumentParser(description=desc)

    verbosity_help = 'Verbosity level (default: %(default)s)'
    choices = [
        logging.getLevelName(logging.DEBUG),
        logging.getLevelName(logging.INFO),
        logging.getLevelName(logging.WARN),
        logging.getLevelName(logging.ERROR)
    ]

    parser.add_argument(
        '-v',
        '--verbosity',
        choices=choices,
        help=verbosity_help,
        default=logging.getLevelName(logging.INFO)
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Set the logging to console level
    logging.basicConfig(level=args.verbosity)

    return args


if __name__ == '__main__':
    main(**parse_args().__dict__)
