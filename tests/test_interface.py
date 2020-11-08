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
import random
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


@patch.object(interface.Com, 'connection', return_value=random.choice([True, False]))
def test_connection(mock_method):
	retr = interface.Com.connection()
	mock_method.assert_called_with()

	assert isinstance(retr, bool)


@patch.object(interface.Com, 'connection', return_value=random.choice([True, False]))
def test_setVible(mock_method1):
	retr = interface.Com.connection()
	mock_method1.assert_called_with()
	method = interface.Com.setVisible(retr)
	visible = interface.check_connection(method)

	assert isinstance(visible, bool)