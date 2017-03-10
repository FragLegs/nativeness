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
MODEL_TYPES = {
    'majority': nativeness.models.majority.Majority,
    'logistic_avg': nativeness.models.logistic.LogisticAvg,
    'logistic_max': nativeness.models.logistic.LogisticMax,
}


# allow instance to not have properly configured tensorflow (for logistic)
try:
    import nativeness.models.pool
    MODEL_TYPES.update({
        'pool_avg': nativeness.models.pool.BiLSTMPoolAvg,
        'pool_max': nativeness.models.pool.BiLSTMPoolMax,
    })
except:
    print('Cannot load tensorflow')


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
        model = load_model(config.results_path)
        evaluate(model)
    else:
        # create model
        model = MODEL_TYPES[config.model_type](config)
        train(model)
        evaluate(model)


def load_model(results_path):
    """
    Loads a model type

    Parameters
    ----------
    results_path : str
        where an existing config might be

    Returns
    -------
    NativenessModel
    """
    # use the results path to get the model data
    config_path = os.path.join(results_path, 'model.config')

    # load the model data
    config = data.load(config_path, as_type='pickle')

    weights_path = os.path.join(results_path, 'model.weights')

    model = MODEL_TYPES[config.model_type](config)
    model.load(weights_path)

    log.info('Loaded a {}'.format(model.__class__.__name__))

    return model


def train(model):
    # save the model config
    log.debug('Saving the model configuration')
    data.save(
        model.config.results_path,
        model.config,
        'model.config',
        as_type='pickle'
    )

    log.debug('Training {} model'.format(model.config.model_type))

    # train model
    dev_preds = np.asarray(
        model.train(
            data.train_generator(model.config),
            data.dev_generator(model.config)
        )
    )

    log.debug('Saving the dev predictions')
    dev_data = data.load(os.path.join(model.config.input_path, 'dev.csv'))

    if model.config.debug:
        dev_data = data.debug(dev_data)

    dev_data['prediction'] = dev_preds
    data.save(
        model.config.results_path,
        dev_data,
        'dev_preds.csv',
        as_type='csv'
    )

    # load dev data
    log.debug('Calculating dev metrics')

    dev_metrics = nativeness.utils.metrics.calculate(dev_data, dev_preds)
    log.info('Dev AUC = {}'.format(dev_metrics['AUC']))
    log.info('Dev Length Correlation = {}'.format(dev_metrics['length_corr']))

    log.debug('Saving metrics')
    data.save(
        model.config.results_path,
        dev_metrics,
        'dev_metrics.json',
        as_type='json'
    )

    # plot the PR curve and save it
    plot_path = os.path.join(model.config.results_path, 'dev_roc_curve.png')
    metrics.plot(
        dev_data.non_native.values,
        dev_preds,
        model.config.model_type,
        plot_path
    )


def evaluate(model):
    log.debug('Predicting on test data')

    # get predictions
    preds, window_preds = model.predict(data.test_generator(model.config))

    # save predictions
    log.debug('Saving the test predictions')
    # load test data
    test_data = data.load(os.path.join(model.config.input_path, 'test.csv'))
    if model.config.debug:
        test_data = data.debug(test_data)

    test_data['prediction'] = preds
    data.save(
        model.config.results_path,
        test_data,
        'test_preds.csv',
        as_type='csv'
    )

    # save the window preds
    window_data = {
        'size': model.config.window_size,
        'stride': model.config.window_stride,
        'preds': [p.tolist() for p in window_preds],
        'used_max': model.__class__.__name__.endswith('Max')
    }
    data.save(
        model.config.results_path,
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
        model.config.results_path,
        test_metrics,
        'test_metrics.json',
        as_type='json'
    )

    # plot the PR curve and save it
    plot_path = os.path.join(model.config.results_path, 'test_roc_curve.png')
    metrics.plot(
        test_data.non_native.values,
        preds,
        model.config.model_type,
        plot_path
    )


def parse_args():
    """
    Parses the arguments from the command line

    Returns
    -------
    argparse.Namespace
    """
    desc = 'Train and eveluate a Nativeness model'
    epilog = (
        'In addition, any parameters in Config can be overridden by adding '
        '"name=value" as an argument. As a shortcut, just adding "name" is '
        'equivalent to "name=True"'
    )
    parser = argparse.ArgumentParser(description=desc, epilog=epilog)

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

    reload_help = (
        'Path to an existing results directory. The model saved there will be '
        'reloaded and no training will be done (only evaluate)'
    )
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

    # drop any extra arguments in an "extra" array
    parser.add_argument('extra', nargs=argparse.REMAINDER)

    # Parse the command line arguments
    args = parser.parse_args()

    # Set the logging to console level
    logging.basicConfig(level=args.verbosity)

    return args


if __name__ == '__main__':
    main(**parse_args().__dict__)
