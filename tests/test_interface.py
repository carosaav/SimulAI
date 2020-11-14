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
from numpy.testing import assert_equal
from simulai import interface
# import random

# ============================================================================
# TESTS
# ============================================================================


@pytest.fixture
def base():
    com = interface.Com("MaterialHandling.spp")

    return com


def test_Com(base):

    assert isinstance(base.model_name, str)
    assert_equal(base.model_name, "MaterialHandling.spp")
    assert isinstance(base.is_connected, bool)
    assert_equal(base.is_connected, False)
    assert isinstance(base.plant_simulation, str)
    assert_equal(base.plant_simulation, "")


def test_get_path_file_model(base):

    assert isinstance(base.get_path_file_model(), str)


def test_connection(base):
    comm = base.connection()

    assert isinstance(comm, bool)


@pytest.mark.xfail
@patch.object(interface.Com, 'connection', return_value=True)
def test_setVisible(base, mock_method):
    valor = interface.Com.connection()
    mock_method.assert_called_with()

    with pytest.raises(ConnectionError):
        base.setVisible(True)
