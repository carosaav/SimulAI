#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   SimulAI Project (https://github.com/carosaav/SimulAI).
# Copyright (c) 2020, Perez Colo Ivo, Pirozzo Manuel Bernardo, Saavedra Sueldo Carolina
# License: MIT
#   Full Text: https://github.com/carosaav


# =============================================================================
# DOCS
# =============================================================================

"""This file is for distribute and install SimulAI
"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import pathlib

from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup


# =============================================================================
# CONSTANTS
# =============================================================================

REQUIREMENTS = ["numpy", "win32", "matplotlib"]

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

with open(PATH / "README.md") as fp:
    LONG_DESCRIPTION = fp.read()

with open(PATH / "simulai" / "__init__.py") as fp:
    for line in fp.readlines():
        if line.startswith("__version__ = "):
            VERSION = line.split("=", 1)[-1].replace('"', '').strip()
            break


DESCRIPTION = "SimulAI - Simulation + Artificial Intelligence"


# =============================================================================
# FUNCTIONS
# =============================================================================

def do_setup():
    setup(
        name="simlai",
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',

        author=[
            "Perez Colo Ivo",
            "Pirozzo Manuel Bernardo",
            "Saavedra Sueldo Carolina"],
        author_email=[
        	"ivoperezcolo@gmail.com",
        	"ber_pirozzo@hotmail.com.ar",
        	"carosaavedrasueldo@gmail.com"]
        url="https://github.com/carosaav/SimulAI",
        license="MIT",

        keywords=["simulai", "simulation", "artificial intelligence", "decision sciences", "optimization"],

        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering"],

        packages=["simulai"],
        py_modules=["ez_setup"],

        install_requires=REQUIREMENTS)


if __name__ == "__main__":
    do_setup()
