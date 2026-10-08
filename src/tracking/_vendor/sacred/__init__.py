#!/usr/bin/env python
# coding=utf-8
"""
The main module of sacred.

It provides access to the two main classes Experiment and Ingredient.
"""

from tracking._vendor.sacred.__about__ import __version__, __author__, __author_email__, __url__
from tracking._vendor.sacred.settings import SETTINGS
from tracking._vendor.sacred.experiment import Experiment
from tracking._vendor.sacred.ingredient import Ingredient
from tracking._vendor.sacred import observers
from tracking._vendor.sacred.host_info import host_info_getter, host_info_gatherer
from tracking._vendor.sacred.commandline_options import cli_option


__all__ = (
    "Experiment",
    "Ingredient",
    "observers",
    "host_info_getter",
    "__version__",
    "__author__",
    "__author_email__",
    "__url__",
    "SETTINGS",
    "host_info_gatherer",
    "cli_option",
)
