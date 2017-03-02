# -*- coding: utf-8 -*-
import argparse
import logging
import os

import pandas as pd

import nativeness.models.base
# import nativeness.models.baseline
import nativeness.models.majority
import nativeness.utils.data as data
import nativeness.utils.metrics as metrics


log = logging.getLogger(name=__name__)


def main(**kwargs):
    """
    Runs the training, prediction and analysis of a nativeness model

    Parameters
    ----------
    **kwargs : keyword arguments
        Any extra arguments get passed on to config
    """
    # set up config
    config = nativeness.models.base.Config(**kwargs)

    # create model
    model_types = {
        # 'baseline': nativeness.models.baseline.Baseline(config),
        'majority': nativeness.models.majority.Majority(config)
    }
    model = model_types[config.model_type]

    # load training data
    train_data = data.load(os.path.join(config.input_path, 'train.csv'))

    # load dev data
    dev_data = data.load(os.path.join(config.input_path, 'dev.csv'))

    # make sure test data is present, but don't load it yet
    assert os.path.exists(os.path.join(config.input_path, 'test.csv'))

    # only use 10 instances if we're debugging
    if config.debug:
        train_data = pd.concat(
            [
                train_data[train_data.non_native].iloc[:5],
                train_data[~train_data.non_native].iloc[:5]
            ]
        )

        dev_data = pd.concat(
            [
                train_data[train_data.non_native].iloc[:5],
                train_data[~train_data.non_native].iloc[:5]
            ]
        )

    # train model
    model.train(train_data, dev_data)

    # load test data
    test_data = data.load(os.path.join(config.input_path, 'test.csv'))

    # get predictions
    preds = model.predict(test_data.text.values)

    # calculate metrics
    m = metrics.calculate(test_data.non_native.values, preds)

    # print metrics
    metrics.show(m)

    # save metrics
    data.save(config.output_path, m, 'metrics', as_type='json')

    # save predictions
    test_data['preds'] = preds
    data.save(config.output_path, test_data, 'preds.csv', as_type='csv')


def parse_args():
    """
    Parses the arguments from the command line

    Returns
    -------
    argparse.Namespace
    """
    desc = 'Train and eveluate a Nativeness model'
    parser = argparse.ArgumentParser(description=desc)

    input_path_help = 'Directory where the data is located'
    parser.add_argument(
        '-i',
        '--input-path',
        type=str,
        default='/research/ella/nativeness',
        help=input_path_help
    )

    output_path_help = 'Directory where the results should be written'
    parser.add_argument(
        '-o',
        '--output-path',
        type=str,
        default='/research/ella/nativeness/results',
        help=output_path_help
    )

    model_type_help = 'Which model to use'
    parser.add_argument(
        '-m',
        '--model-type',
        type=str,
        default='majority',
        help=model_type_help
    )

    debug_help = 'Whether to run on a limited set of data'
    parser.add_argument(
        '-d',
        '--debug',
        action='store_true',
        help=debug_help
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
