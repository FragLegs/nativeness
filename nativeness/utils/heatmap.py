# -*- coding: utf-8 -*-
"""
This module is not original work.
It is modified from a set of tools that we use at Turnitin.
"""
import cgi
import logging
import math

import numpy as np

log = logging.getLogger(name=__name__)


def html_sanitize(text):
    sane = cgi.escape(text)
    sane = sane.replace("\n", "<br>")
    return sane


def html_heatmap(text,
                 spans,
                 title='A heatmap',
                 min_value=None,
                 max_value=None,
                 lens=lambda heat: math.pow(heat, 2),
                 hue=50):
    """
    text: str
        plaintext of the whole essay
    spans: list of triples
        [(star_index, end_index, value), ...]
    min_value: float
        the minimum value for "heat" -- will be completely white
    max_value: float
        the maximum value for "heat" -- will be as bright as possible
    lens: function
        transform the normalized heat value to make it pretty.
    hue: int
        0-360. I find that 50 is a pleasant shade of gold

    Returns
    -------
    str
        The html
    """
    out = []
    out.append('<h3>{}</h3>'.format(title))

    epsilon = 0.00001

    values = zip(*spans)[2]
    if min_value is None:
        min_value = min(values) - epsilon
    if max_value is None:
        max_value = max(values) + epsilon

    last_end = 0
    for start, end, value in spans:
        if last_end < start:
            border_text = html_sanitize(text[last_end:start])
            out.append(border_text)

        heat = (value - min_value) / float(max_value - min_value)

        assert not np.isnan(heat), value

        # lightness values less than 50% tend to be unreadable.
        lightness = 100 - int(50 * lens(heat))
        span_text = html_sanitize(text[start:end])
        span_html = (
            '<span style="background-color:hsl({}, 100%, {}%)">{}</span>'
        )
        span_html = span_html.format(hue, lightness, span_text)

        out.append(span_html)

        last_end = end

    if last_end < len(text):
        out.append(html_sanitize(text[last_end:len(text)]))

    return ''.join(out)
