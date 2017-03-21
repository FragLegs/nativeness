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


def main(results_path, **kwargs):
    # read in the window preds
    log.debug('Loading predictions')
    window_path = os.path.join(results_path, 'test_window_preds.json')
    window_data = nativeness.utils.data.load(window_path, as_type='json')
    window_size = window_data['size']
    window_stride = window_data['stride']
    window_preds = window_data['preds']
    used_max = window_data['used_max']

    log.debug('Determining cutoffs')
    all_preds = np.array(window_preds)
    log.debug(type(all_preds))
    log.debug(all_preds.shape)
    all_preds = np.concatenate(all_preds)
    log.debug(all_preds.shape)
    all_preds = np.sort(all_preds)
    low = all_preds[10000]
    high = all_preds[-100]

    n = len(all_preds)
    mid = n / 2
    m1 = all_preds[mid - 50]
    m2 = all_preds[mid + 50]

    log.debug('low = {}'.format(low))
    log.debug('mid is {} to {}'.format(m1, m2))
    log.debug('high = {}'.format(high))

    del(all_preds)

    # load the essays
    log.debug('Loading essays')
    essays_path = os.path.join(results_path, 'test_preds.csv')
    essays = nativeness.utils.data.load(essays_path)

    log.debug('LOW')
    make_csv(
        essays,
        window_preds,
        window_size,
        window_stride,
        low=0.0,
        high=low,
        path=os.path.join(results_path, 'window_preds_low.csv')
    )


    log.debug('MEDIUM')
    make_csv(
        essays,
        window_preds,
        window_size,
        window_stride,
        low=m1,
        high=m2,
        path=os.path.join(results_path, 'window_preds_medium.csv')
    )

    log.debug('HIGH')
    make_csv(
        essays,
        window_preds,
        window_size,
        window_stride,
        low=high,
        high=1.0,
        path=os.path.join(results_path, 'window_preds_high.csv')
    )


def make_csv(essays, preds, size, stride, low, high, path):

    prog = nativeness.utils.progress.Progbar(target=len(essays))
    i = 0

    recs = []

    log.debug('Generating windows')
    # for each essay
    for row_id, row in essays.iterrows():
        text = row.text
        window_preds = preds[i]
        base = {
            'essay_id': row_id, 'score': row.prediction, 'ELL': row.non_native
        }
        recs.extend(
            make_windows(text, window_preds, size, stride, base, low, high)
        )

        i += 1
        prog.update(i)

    log.debug('Making DataFrame')
    df = pd.DataFrame.from_records(recs)

    log.debug('THere are {} unique values'.format(len(set(df.window_pred.values))))

    log.debug('Sorting DataFrame')
    df.sort_values('window_pred', ascending=True, inplace=True)

    log.debug('Saving DataFrame with {} windows'.format(len(df)))
    df.to_csv(path, encoding='utf8')

    log.info('Data saved at {}'.format(path))

    print('}\n\\texttt{\\fontsize{.28cm}{.1cm}\\selectfont \\frenchspacing '.join(df.sample(50).text.values))
    print('-'*100)


def make_windows(text, preds, size, stride, base, low, high):
    ret = []

    for i, p in enumerate(preds):
        if (p < low) or (p > high):
            continue
        d = copy.copy(base)
        d['window_pred'] = p
        start = i * stride
        d['text'] = text[start: start + size]
        ret.append(d)

    return ret


def parse_args():
    """
    Parses the arguments from the command line

    Returns
    -------
    argparse.Namespace
    """
    desc = 'Collect top windows'
    parser = argparse.ArgumentParser(description=desc)

    results_path_help = 'Where the results data lives'
    parser.add_argument(
        'results_path',
        type=str,
        help=results_path_help
    )

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
