
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
from simulai import sim


# ============================================================================
# TESTS
# ============================================================================


@pytest.fixture
def var_input():
    frame = "Modelo"
    espera = sim.DiscreteVariable(
        "Espera", 60, 300, 10, "Models." + frame + ".espera")
    stock = sim.DiscreteVariable(
        "Stock", 10, 50, 10, "Models." + frame + ".stock")
    numviajes = sim.DiscreteVariable(
        "Numero de viajes", 1, 5, 1, "Models." + frame + ".numviajes")
    vi = [espera, stock, numviajes]

    return vi


@pytest.fixture
def var_out():
    frame = "Modelo"
    transportes = sim.OutcomeVariable(
        "Distancia Transportes", "Models." + frame + ".transportes", 2, 9)
    buffers = sim.OutcomeVariable(
        "Llenado buffers", "Models." + frame + ".buffers", 3, 20)
    salidas = sim.OutcomeVariable(
        "Espera en las Salidas", "Models." + frame + ".salidas", 2, 20)
    vo = [transportes, buffers, salidas]

    return vo


@pytest.fixture
def base(var_input, var_out):
    plant = sim.BasePlant(
        method=sim.Qlearning(v_i=var_input, episodes_max=1, steps_max=10),
        v_i=var_input,
        v_o=var_out,
        filename="MaterialHandling.spp",
        modelname="Model",
    )

    return plant


@pytest.fixture
def my_method(var_input):
    method = sim.Qlearning(v_i=var_input, episodes_max=1, steps_max=10)

    return method

testdata = [(0.1, 0.2, "string", "string"), 
            ((4.5 + 3j), (4.5 + 3j), 0.1, 0.2), 
            (False, True, (4.5 + 3j), (4.5 + 3j)), 
            ([1, "test", 2], [1, "test", 2], False, True), 
            (("a", "b", "c"), ("a", "b", "c"), 
                [1, "test", 2], [1, "test", 2]), 
            ({"a":1, "b":"test", "c":2}, {"a":1, "b":"test", "c":2}, 
                ("a", "b", "c"), ("a", "b", "c") ), 
            (set([3.0, 'Car', True]), set([3.0, 'Car', True]), 
                {"a":1, "b":"test", "c":2}, {"a":1, "b":"test", "c":2}),
            (0.1, 0.2, set([3.0, 'Car', True]), set([3.0, 'Car', True]))]


def test_DiscreteVariable():
    """Test that the arguments that define a discrete variable
    are of the right type.
    """
    parm = sim.DiscreteVariable("Time", 0, 10, 1, "path")

    assert isinstance(parm.name, str)
    assert isinstance(parm.lower_limit, int)
    assert isinstance(parm.upper_limit, int)
    assert isinstance(parm.step, int)
    assert isinstance(parm.path, str)

    with pytest.raises(TypeError):
        sim.DiscreteVariable(testdata)

def test_OutcomeVariable():
    """Test that the output variable has the correct types of arguments."""
    parm = sim.OutcomeVariable("Time", "path", 5, 1)

    assert isinstance(parm.name, str)
    assert isinstance(parm.path, str)
    assert isinstance(parm.column, int)
    assert isinstance(parm.num_rows, int)

    with pytest.raises(TypeError):
        sim.OutcomeVariable(testdata)


def test_BasePlant(base):
    """Test data type of argument"""
    assert isinstance(base.v_i, list)
    assert isinstance(base.v_o, list)
    assert isinstance(base.filename, str)
    assert isinstance(base.modelname, str)

    with pytest.raises(TypeError):
        sim.OutcomeVariable(testdata)


def test_get_file_name_plant(base):
    """Test data type and value of file name"""
    filename = base.get_file_name_plant()

    assert filename == "MaterialHandling.spp"
    assert isinstance(filename, str)


@patch.object(sim.BasePlant, 'update', return_value=1.10)
def test_update(mock_method):
    sim.BasePlant.update([60, 10, 1])
    mock_method.assert_called_with([60, 10, 1])


@patch.object(sim.BasePlant, 'process_simulation')
def test_process_simulation(mock_method2):
    """Test that the connection() function returns a boolean type value.

    Use the mock of the simulation software.
    """
    sim.BasePlant.process_simulation()
    mock_method2.assert_called_with()


def test_default_Q(my_method):
    """Test the data type of the Qlearning function arguments and

    check the default values.
    """
    assert isinstance(my_method.s, list)
    assert isinstance(my_method.a, list)
    assert isinstance(my_method.v_i, list)
    assert isinstance(my_method.alfa, float)
    assert isinstance(my_method.gamma, float)
    assert isinstance(my_method.epsilon, float)
    assert isinstance(my_method.episodes_max, int)
    assert isinstance(my_method.steps_max, int)
    assert isinstance(my_method.r_episode, np.ndarray)
    assert_equal(my_method.s, [])
    assert_equal(my_method.a, [])
    assert_equal(my_method.alfa, 0.10)
    assert_equal(my_method.gamma, 0.90)
    assert_equal(my_method.epsilon, 0.10)
    assert_equal(my_method.episodes_max, 1)
    assert_equal(my_method.steps_max, 10)


def test_arrays(my_method):
    """Test the data type of the arrays generated with the limit and

    step information of the input variables.
    """
    my_method.arrays()
    assert_equal(len(my_method.s), 3)
    assert_equal(len(my_method.a), 3)


def test_ini_saq(my_method):
    """Test that the output Q matrix has the necessary characteristics.

    Initially the data type is checked.
    Then dimensions and composition are checked.
    """
    Q, S, A = my_method.ini_saq()

    assert isinstance(Q, np.ndarray)
    assert isinstance(S, np.ndarray)
    assert isinstance(A, np.ndarray)
    assert Q.shape == (625, 27)
    assert np.all((Q == 0)) == True
    assert S.shape == (625, 3)
    assert np.all((S == 0)) == False
    assert A.shape == (27, 3)
    assert np.all((A == 0)) == False


def test_choose_action(my_method):
    """Test that the function choose_action takes the row "0" of the

    Q array when p < 1 - epsilon or takes a random row otherwise.
    """
    method = my_method
    QSA = method.ini_saq()
    i = method.choose_action(np.random.randint(624))

    # assert (isinstance(i, int))
    assert_equal(i, 0)


@patch.object(sim.Qlearning, 'process')
def test_process(mock_method3):
    """Test that the process function returns an array.

    Use the mock of the simulation software (subscriber).
    """
    sim.Qlearning.process()
    mock_method3.assert_called_with()
