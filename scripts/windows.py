# -*- coding: utf-8 -*-
import argparse
import copy
import logging
import os

import numpy as np
import pandas as pd

import nativeness.utils.data
import nativeness.utils.progress

log = logging.getLogger(name=__name__)


def main(**kwargs):

    path = '/research/ella/nativeness/final_results'
    results = [
        'logistic_nn_avg/20170315_024847',
        'logistic_nn_max/20170315_132419',
        'pool_avg/20170314_200044',
        'pool_max/20170315_135127',
        'prompt_avg/20170314_034434',
        'prompt_max/20170316_131749'
    ]

    wp = {}
    for r in results:
        results_path = os.path.join(path, r)
        window_path = os.path.join(results_path, 'test_window_preds.json')
        window_data = nativeness.utils.data.load(window_path, as_type='json')
        window_size = window_data['size']
        window_stride = window_data['stride']
        window_preds = window_data['preds']
        wp[r.split('/')[0]] = window_preds

    # load the essays
    log.debug('Loading essays')
    essays_path = os.path.join(results_path, 'test_preds.csv')
    essays = nativeness.utils.data.load(essays_path)

    windows = []
    scores = {r: [] for r in wp.keys()}

    # randomly select 200 essays
    for i in np.random.randint(0, len(essays), 200):
        j = np.random.randint(0, len(window_preds[i]))
        start = j * window_stride
        end = start + window_size

        windows.append(essays.iloc[i].text[start:end])
        for r in wp.keys():
            scores[r].append(wp[r][i][j])

    wdf = pd.DataFrame()
    wdf['annotation'] = np.nan
    wdf['text'] = windows
    wdf.to_csv(os.path.join(path, 'score_me.csv'), encoding='utf8')

    sdf = pd.DataFrame()
    for r, s in scores.iteritems():
        sdf[r] = s
    sdf.to_csv(os.path.join(path, 'scores.csv'), encoding='utf8')


def parse_args():
    """
    Parses the arguments from the command line

    Returns
    -------
    argparse.Namespace
    """
    desc = 'Collect random windows'
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
