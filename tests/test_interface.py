# This file is part of the
#   SimulAI Project (https://github.com/carosaav/SimulAI).
# Copyright (c) 2020, Perez Colo Ivo, Pirozzo Bernardo Manuel,
# Carolina Saavedra Sueldo
# License: MIT
#   Full Text: https://github.com/carosaav/SimulAI/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================


import pytest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_equal
from simulai import interface


# ============================================================================
# TESTS
# ============================================================================


@pytest.fixture
def base():
	com = interface.Com("frame") 

	return com


def test_Com(base):
		
	assert isinstance(base.model_name, str)
	assert isinstance(base.is_connected, bool)


def test_get_path_file_model(base):
 
    assert isinstance(base.get_path_file_model(), str)

def test_connection(base):
	