# -*- coding: utf-8 -*-
import argparse
import logging
import os

import numpy as np

import nativeness.models.base
import nativeness.models.logistic
import nativeness.models.majority
import nativeness.utils.data as data
import nativeness.utils.metrics as metrics
import nativeness.utils.progress


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

    # make sure test data is present, but don't load it yet
    assert os.path.exists(os.path.join(config.input_path, 'test.csv'))

    if config.reload is not None:
        model = data.load(config.reload, as_type='pickle')
        evaluate(model, config)
    else:
        # create model
        model_types = {
            # 'baseline': nativeness.models.baseline.Baseline,
            'majority': nativeness.models.majority.Majority,
            'logistic': nativeness.models.logistic.Logistic,
        }
        model = model_types[config.model_type](config)
        train(model, config)
        evaluate(model, config)


def train(model, config):
    log.debug('Training {} model'.format(config.model_type))

    # train model
    dev_preds = np.asarray(
        model.train(data.train_generator(config), data.dev_generator(config))
    )

    log.debug('Saving the dev predictions')
    dev_data = data.load(os.path.join(config.input_path, 'dev.csv'))

    if config.debug:
        dev_data = data.debug(dev_data)

    dev_data['prediction'] = dev_preds
    data.save(config.results_path, dev_data, 'dev_preds.csv', as_type='csv')

    # save the model
    log.debug('Saving the model')
    data.save(config.results_path, model, 'model', as_type='pickle')

    # load dev data
    log.debug('Calculating dev metrics')

    dev_metrics = nativeness.utils.metrics.calculate(dev_data, dev_preds)
    log.info('Dev AUC = {}'.format(dev_metrics['AUC']))
    log.info('Dev Length Correlation = {}'.format(dev_metrics['length_corr']))

    log.debug('Saving metrics')
    data.save(
        config.results_path, dev_metrics, 'dev_metrics.json', as_type='json'
    )

    # plot the PR curve and save it
    plot_path = os.path.join(config.results_path, 'dev_roc_curve.png')
    metrics.plot(
        dev_data.non_native.values, dev_preds, config.model_type, plot_path
    )


def evaluate(model, config):
    log.debug('Predicting on test data')

    # get predictions
    preds, window_preds = model.predict(data.test_generator(config))

    # save predictions
    log.debug('Saving the test predictions')
    # load test data
    test_data = data.load(os.path.join(config.input_path, 'test.csv'))
    if config.debug:
        test_data = data.debug(test_data)

    test_data['prediction'] = preds
    data.save(config.results_path, test_data, 'test_preds.csv', as_type='csv')

    # save the window preds
    window_data = {
        'size': config.window_size,
        'stride': config.window_stride,
        'preds': [p.tolist() for p in window_preds],
    }
    data.save(
        config.results_path,
        window_data,
        'test_window_preds.json',
        as_type='json'
    )

    # calculate metrics
    log.debug('Calculating test metrics')
    test_metrics = metrics.calculate(test_data, preds)

    # save metrics
    log.debug('Saving metrics')
    data.save(
        config.results_path, test_metrics, 'test_metrics.json', as_type='json'
    )

    # plot the PR curve and save it
    plot_path = os.path.join(config.results_path, 'test_roc_curve.png')
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
