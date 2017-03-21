# -*- coding: utf-8 -*-
import argparse
import logging
import os
import SimpleHTTPServer
import threading

import numpy as np

import nativeness.utils.data
import nativeness.utils.heatmap
import nativeness.utils.progress

log = logging.getLogger(name=__name__)


def main(input_path, results_path, serve, max_only, **kwargs):
    """
    Use the window preds to get intensities for each character and save it
    as HTML

    Parameters
    ----------
    input_path : str
        Directory where the data is located
    results_path : str
        Directory where the test results are located
    serve : bool
        Whether to open a server
    max_only : bool
        Only highlight the most ELL-like window
    **kwargs : keyword args
        ignored
    """
    # load the window data
    log.debug('Loading predictions')
    window_path = os.path.join(results_path, 'test_window_preds.json')
    window_data = nativeness.utils.data.load(window_path, as_type='json')
    window_size = window_data['size']
    window_stride = window_data['stride']
    window_preds = window_data['preds']
    used_max = window_data['used_max']

    # load the essays
    log.debug('Loading essays')
    essays_path = os.path.join(results_path, 'test_preds.csv')
    essays = nativeness.utils.data.load(essays_path)

    # load the config
    config = nativeness.utils.data.load(
        os.path.join(results_path, 'model.config'), as_type='pickle'
    )

    # create an html directory
    html_path = os.path.join(
        results_path, 'html' + ('_max' if max_only else '')
    )

    try:
        os.mkdir(html_path)
    except:
        pass

    prog = nativeness.utils.progress.Progbar(target=len(essays))
    i = 0

    log.debug('Creating HTML')
    # for each essay
    for row_id, row in essays.iterrows():
        # make filename
        file_path = os.path.join(
            html_path,
            '{0:0.10f}_{1}_{2}.html'.format(
                row.prediction,
                'ELL' if row.non_native else 'ENG',
                row_id
            )
        )

        # calculate character intensities
        char_intensities = calculate_intensities(
            preds=window_preds[i],
            essay_length=len(row.text),
            size=window_size,
            stride=window_stride,
            used_max=used_max,
            max_only=max_only
        )

        title = (
            '{}<br><br>Is ELL? {}'
            '<br>P(ELL) = {}'
            '<br>Model: {}'
        )
        title = title.format(
            row.uid, row.non_native, row.prediction, config.model_type
        )

        # generate html
        html = nativeness.utils.heatmap.html_heatmap(
            text=row.text,
            spans=char_intensities,
            title=title,
            # min_value=0.0,
            # max_value=1.0
        )

        # save html
        with open(file_path, 'w') as fout:
            fout.write(html)

        i += 1
        prog.update(i)

    # open server, if desired
    if serve:
        open_server(html_path)
    else:
        print('HTML written to {}'.format(html_path))


def calculate_intensities(preds,
                          essay_length,
                          size,
                          stride,
                          used_max,
                          max_only):
    """
    Averages windows per character

    Parameters
    ----------
    preds : iterable of float
        The predictions for each window
    essay_length : int
        How long the text is
    size : int
        How big each window is
    stride : int
        How many characters between each window
    used_max : bool
        If the model used the maximum window prediction, then we only want
        to visualize that window
    max_only : bool
        Only show the maximum window prediction

    Returns
    -------
    list of triples
        value = average prediction
        [(star_index, end_index, value), ...]
    """
    values = np.zeros(essay_length, dtype=np.float)
    counts = np.zeros(essay_length, dtype=np.float)

    max_pred = np.max(preds)
    for i, pred in enumerate(preds):
        if max_only and pred < max_pred:
            continue
        start = i * stride
        values[start: start + size] += pred
        counts[start: start + size] += 1.0

    # because of the stride, not all characters get predictions
    counts[counts == 0.0] += 1.0

    values = values / counts

    assert not np.any(np.isnan(values)), (counts == 0.0).sum()

    return [(i, i + 1, v) for i, v in enumerate(values)]


def open_server(html_path):
    """
    Open a server to serve the html

    Parameters
    ----------
    html_path : str
        Where the html lives
    """
    PORT = 8000

    os.chdir(html_path)

    Handler = SimpleHTTPServer.SimpleHTTPRequestHandler
    httpd = SimpleHTTPServer.BaseHTTPServer.HTTPServer(('', PORT), Handler)

    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    print('Serving at http://localhost:{}'.format(PORT))

    try:
        while True:
            pass
    except KeyboardInterrupt:
        httpd.shutdown()


def parse_args():
    """
    Parses the arguments from the command line

    Returns
    -------
    argparse.Namespace
    """
    desc = 'Visualize the essays'
    parser = argparse.ArgumentParser(description=desc)

    input_path_help = 'Directory where the data is located'
    parser.add_argument(
        '-i',
        '--input-path',
        type=str,
        default='/research/ella/nativeness',
        help=input_path_help
    )

    results_path_help = 'Directory where the test results are located'
    parser.add_argument(
        'results_path',
        type=str,
        help=results_path_help
    )

    serve_help = 'If this flag is present, open a server to serve the html'
    parser.add_argument(
        '--serve',
        action='store_true',
        help=serve_help
    )

    max_only_help = (
        'If this flag is present, only highlight the most ELL-like window'
    )
    parser.add_argument(
        '--max-only',
        action='store_true',
        help=max_only_help
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
