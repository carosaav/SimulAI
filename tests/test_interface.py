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
from unittest import mock
from numpy.testing import assert_equal
from simulai import interface


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
@mock.patch('simulai.interface.Com.connection',
            mock.MagicMock(return_value=True))
def test_check_connection(base):
    interface.Com("MaterialHandling.spp")
    base.is_connected = base.connection()
    interface.Com.setVisible(True)


@pytest.fixture
def setVisible(value=True):
    pass


@pytest.mark.xfail
def test_wrapper(base, setVisible):
    interface.Com("MaterialHandling")
    base.is_connected = True
    setV = base.setVisible
    check = interface.check_connection(setV)
    r = check.wrapper()

    assert_equal(r, True)
