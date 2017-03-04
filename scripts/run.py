# -*- coding: utf-8 -*-
import argparse
import logging
import os

import pandas as pd

import nativeness.models.base
# import nativeness.models.baseline
import nativeness.models.logistic
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

    if config.reload is not None:
        model = data.load(config.reload, as_type='pickle')
    else:
        # create model
        model_types = {
            # 'baseline': nativeness.models.baseline.Baseline,
            'majority': nativeness.models.majority.Majority,
            'logistic': nativeness.models.logistic.Logistic,
            'logistic_windows': nativeness.models.logistic.LogisticWindows,
        }
        model = model_types[config.model_type](config)

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

    log.debug('Training {} model'.format(config.model_type))

    # train model
    model.train(train_data, dev_data)

    log.debug('Saving the model')

    # save the model
    data.save(config.results_path, model, 'model', as_type='pickle')

    # load test data
    test_data = data.load(os.path.join(config.input_path, 'test.csv'))

    log.debug('Predicting on test data')

    # get predictions
    preds = [
        model.predict(nativeness.utils.text.windows(essay))
        for essay in test_data.text.values
    ]

    # calculate metrics
    m = metrics.calculate(
        test_data.non_native.values, preds, test_data.text.values
    )

    # print metrics
    metrics.show(m)

    # save metrics
    data.save(config.results_path, m, 'metrics', as_type='json')

    # save predictions
    test_data['preds'] = preds
    data.save(config.results_path, test_data, 'preds.csv', as_type='csv')

    # plot the PR curve and save it
    plot_path = os.path.join(config.results_path, 'pr_curve.png')
    metrics.plot(
        test_data.non_native.values, preds, config.model_type, plot_path
    )


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

    reload_help = 'Reload a saved model'
    parser.add_argument(
        '-r',
        '--reload',
        type=str,
        default=None,
        help=reload_help
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
