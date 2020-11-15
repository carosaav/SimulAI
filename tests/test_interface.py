
# This file is part of the
#   SimulAI Project (https://github.com/carosaav/SimulAI).
# Copyright (c) 2020, Perez Colo Ivo, Pirozzo Bernardo Manuel,
# Carolina Saavedra Sueldo
# License: MIT
#   Full Text: https://github.com/carosaav/SimulAI/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================


import random
import pytest
from unittest import mock
from unittest.mock import patch
from numpy.testing import assert_equal
from simulai import interface


# ============================================================================
# TESTS
# ============================================================================


@pytest.fixture
def base():
    com = interface.Com("MaterialHandling.spp")

    return com


@pytest.fixture
def base2():
    com = interface.Com("aaa")

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
    assert isinstance(base.plant_simulation, str)


@patch.object(interface.Com, "connection",
              return_value=random.choice([True, False]))
def test_connection2(mock_method):
    value = interface.Com.connection()
    mock_method.assert_called_once_with()

    assert isinstance(value, bool)


@mock.patch('simulai.interface.Com.connection',
            mock.MagicMock(return_value=True))
def test_check_connection(base):
    value2 = base.connection()

    assert isinstance(value2, bool)


def test_invalidModel(base2):
    with pytest.raises(Exception):
        base.connection()


@patch.object(interface.Com, "setVisible", spec_set=True)
def test_setVisible(mock_method):
    interface.Com.setVisible(True)
    mock_method.assert_called_with(True)


def test_setVisible2(base2):
    with pytest.raises(interface.ConnectionError):
        base2.setVisible()


def test_setValue(base2):
    with pytest.raises(interface.ConnectionError):
        base2.setValue()


@patch.object(interface.Com, "startSimulation", spec_set=True)
def test_startSimulation(mock_method):
    interface.Com.startSimulation(".Models.Modelo")
    mock_method.assert_called_with(".Models.Modelo")
