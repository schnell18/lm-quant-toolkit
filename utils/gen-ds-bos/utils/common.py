#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common functions to serve other packages.

This module includes common functions such as:

    * configuration
    * date arithmetic
    * generic repository access

"""

import configparser
import sys


def eprint(*args, **kwargs):
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


def load_config_item(section, key, default=None):
    """Load the config item from config file.

    Parameters
    ----------
    section: str


    Returns
    -------
    str
        the github access token
    """
    parser = configparser.ConfigParser()
    parser.read("settings.ini")
    if section in parser:
        section = parser[section]
        return section.get(key, default)
    else:
        return default
