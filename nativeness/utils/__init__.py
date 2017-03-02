# -*- coding: utf-8 -*-
import logging


log = logging.getLogger(name=__name__)


class Config(object):
    """
    Stores configuration info for model building
    """
    essays_path = '/research/ella/rivendell/all_essays.csv'

