
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
def var_input(espera, stock, numviajes):
    vi = [espera, stock, numviajes]

    return vi


@pytest.fixture
def espera():
    espera = sim.DiscreteVariable(
        "Espera", 60, 300, 10, "Models.Modelo.espera")
    return espera


@pytest.fixture
def stock():
    stock = sim.DiscreteVariable(
        "Stock", 10, 50, 10, "Models.Modelo.stock")
    return stock


@pytest.fixture
def numviajes():
    numviajes = sim.DiscreteVariable(
        "Numero de viajes", 1, 5, 1, "Models.Modelo.numviajes")
    return numviajes


@pytest.fixture
def transportes():
    transportes = sim.OutcomeVariable(
        "Distancia Transportes", "Models.Modelo.transportes", 2, 9)
    return transportes


@pytest.fixture
def buffers():
    buffers = sim.OutcomeVariable(
        "Llenado buffers", "Models.Modelo.buffers", 3, 20)
    return buffers


@pytest.fixture
def salidas():
    salidas = sim.OutcomeVariable(
        "Espera en las Salidas", "Models.Modelo.salidas", 2, 20)
    return salidas


@pytest.fixture
def var_out(transportes, buffers, salidas):            
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
def BaseM(var_input):
    method = sim.Qlearning(v_i=var_input, episodes_max=1,
                           steps_max=10, seed=None)

    return method


@pytest.fixture
def my_method_S(var_input, request):
    # seed = request
    method = sim.Sarsa(v_i=var_input, episodes_max=1, steps_max=10, seed=None)

    return method


@pytest.mark.parametrize('namef, lowf, upf, stf, pathf', [
                    ("Espera", 60., 300, 10, "Models.Modelo.espera"),
                    ("Espera", 60, 300., 10, "Models.Modelo.espera"),
                    ("Espera", 60, 300, 10., "Models.Modelo.espera"),
                    (["Espera"], 60, 300, 10, "Models.Modelo.espera"),
                    ({"e":"Espera"}, 60, 300, 10, "Models.Modelo.espera"),
                    ("Espera", 60, 300, (4.5 + 3j), "Models.Modelo.espera"),
                    ("Espera", 60, 300, 10, False)])
def test_DiscreteVariable(namef, lowf, upf, stf, pathf):
    """Test that the arguments that define a discrete variable
    are of the right type.
    """
    parm = sim.DiscreteVariable("Espera", 60, 300, 10, "Models.Modelo.espera")

    assert isinstance(parm.name, str)
    assert isinstance(parm.lower_limit, int)
    assert isinstance(parm.upper_limit, int)
    assert isinstance(parm.step, int)
    assert isinstance(parm.path, str)

    with pytest.raises(TypeError):
        sim.DiscreteVariable(namef, lowf, upf, stf, pathf)


@pytest.mark.parametrize('namef, pathf, colf, rowf', [
                                (False, "Model", 2, 9),
                                ("Distance", "Model", 2., 9),
                                ("Distance", True, 2, 9.),
                                (4.2, "Model", 2, 9),
                                ("Distance", {"m":"Model"}, 2, 9)])
def test_OutcomeVariable(namef, pathf, colf, rowf):
    """Test that the output variable has the correct types of arguments."""
    parm = sim.OutcomeVariable("Time", "path", 5, 1)

    assert isinstance(parm.name, str)
    assert isinstance(parm.path, str)
    assert isinstance(parm.column, int)
    assert isinstance(parm.num_rows, int)

    with pytest.raises(TypeError):
        sim.OutcomeVariable(namef, pathf, colf, rowf)


@pytest.mark.parametrize('vif, vof, filenamef, modelnamef', [
    ([espera, stock, numviajes], 
    					[transportes, buffers, salidas], 2, "frame"),
    ([espera, stock, numviajes], 
    					[transportes, buffers, salidas], "MH.spp", 2.),
    (2, [transportes, buffers, salidas], "MH.spp", "frame"),
    ([espera, stock, numviajes], True, "MH.spp", "frame"),
    ("espera, stock, numviajes", 
    					[transportes, buffers, salidas], "MH.spp", "frame")])
def test_BasePlant(base, vif, vof, filenamef, modelnamef):
    """Test data type of argument"""
    assert isinstance(base.v_i, list)
    assert isinstance(base.v_o, list)
    assert isinstance(base.filename, str)
    assert isinstance(base.modelname, str)

    with pytest.raises(TypeError):
        sim.BasePlant(vif, vof, filenamef, modelnamef)


def test_get_file_name_plant(base):
    """Test data type and value of file name"""
    filename = base.get_file_name_plant()

    assert filename == "MaterialHandling.spp"
    assert isinstance(filename, str)


@patch.object(sim.BasePlant, 'update', return_value=np.random.uniform(0, 5))
def test_update(mock_method):
    value = sim.BasePlant.update([60, 10, 1])
    mock_method.assert_called_with([60, 10, 1])

    assert isinstance(value, float)


@pytest.mark.parametrize('vi5', ["espera", "stock", "numviajes", 
                                   "tiempo", "velocidad"])
@pytest.mark.parametrize('vifail, epfail, stfail', [
                                ([espera, stock, numviajes], "ten", 1),
                                ([espera, stock, numviajes], 5, "two"),
                                ([espera, stock, numviajes], 5., 3.),
                                ([espera, stock, numviajes], 5, {"a": "two"}),
                                (1, 5, 3),
                                ("list", 2, 6),
                                ({"a": espera, "b": stock}, 1, 1)])
@pytest.mark.parametrize('var_input, epmax, stmax', [
                                ([espera, stock, numviajes], 1, 10)])
@patch.multiple(sim.BaseMethod, __abstractmethods__=set())
def test_BaseMethod(var_input, vifail, vi5, epmax, epfail, stmax, stfail):
     BaseM = sim.BaseMethod(v_i=var_input, episodes_max=epmax, steps_max=stmax,
                          alfa=0.1, gamma=0.9, epsilon=0.1, 
                                   s=["a", "b"], a=["a", "b"], seed=None)

     assert isinstance(BaseM.s, list)
     assert isinstance(BaseM.a, list)
     assert isinstance(BaseM.v_i, list)
     assert isinstance(BaseM.alfa, float)
     assert isinstance(BaseM.gamma, float)
     assert isinstance(BaseM.epsilon, float)
     assert isinstance(BaseM.episodes_max, int)
     assert isinstance(BaseM.steps_max, int)
     assert isinstance(BaseM.r_episode, np.ndarray)
     assert_equal(len(BaseM.s), 2)
     assert_equal(len(BaseM.a), 2)
     assert_equal(BaseM.alfa, 0.10)
     assert_equal(BaseM.gamma, 0.90)
     assert_equal(BaseM.epsilon, 0.10)
     assert_equal(BaseM.episodes_max, 1)
     assert_equal(BaseM.steps_max, 10)

     with pytest.raises(TypeError):
          sim.BaseMethod (vifail, epfail, stfail)

     with pytest.raises(Exception):
          sim.BaseMethod (vi5, epfail, stfail)

@pytest.mark.xfail  
def test_Q():
    method = sim.Qlearning()

    assert isinstance(method.v_i, list)
    assert isinstance(method.episodes_max, int)
    assert isinstance(method.steps_max, int)

    with pytest.raises(TypeError):
        sim.Qlearning(vifail, epfail, stfail)


def test_arrays(BaseM):
    """Test the data type of the arrays generated with the limit and

    step information of the input variables.
    """
    BaseM.arrays()
    assert_equal(len(BaseM.s), 3)
    assert_equal(len(BaseM.a), 3)


def test_ini_saq(BaseM):
    """Test that the output Q matrix has the necessary characteristics.

    Initially the data type is checked.
    Then dimensions and composition are checked.
    """
    Q, S, A = BaseM.ini_saq()

    assert isinstance(Q, np.ndarray)
    assert isinstance(S, np.ndarray)
    assert isinstance(A, np.ndarray)
    assert Q.shape == (625, 27)
    assert S.shape == (625, 3)
    assert A.shape == (27, 3)
    assert (Q == 0).all()
    assert bool((S == 0).all()) is False
    assert bool((A == 0).all()) is False


@pytest.mark.xfail
@pytest.mark.parametrize('input_seed, expected',
                         [(24, 0), (20, 0), (12, 0)],
                         indirect=['input_seed'])
def test_choose_action(BaseM, input_seed, expected):
    """Test that the function choose_action takes the row "0" of the

    Q array when p < 1 - epsilon or takes a random row otherwise.
    """
    method = BaseM(seed=input_seed)
    method.ini_saq()
    i = method.choose_action(np.random.randint(624))

    # assert (isinstance(i, tuple))
    assert_equal(i, expected)


@patch.object(sim.Qlearning, 'process', return_value=[1., 0., 0., 2., 2.])
def test_process(mock_method2):
    """Test that the process function returns an array."""
    r = sim.Qlearning.process()
    mock_method2.assert_called_with()

    assert isinstance(r, list)


def test_default_Sarsa(my_method_S):
    """Test the data type of the Sarsa function arguments and

    check the default values.
    """
    assert isinstance(my_method_S.s, list)
    assert isinstance(my_method_S.a, list)
    assert isinstance(my_method_S.v_i, list)
    assert isinstance(my_method_S.alfa, float)
    assert isinstance(my_method_S.gamma, float)
    assert isinstance(my_method_S.epsilon, float)
    assert isinstance(my_method_S.episodes_max, int)
    assert isinstance(my_method_S.steps_max, int)
    assert isinstance(my_method_S.r_episode, np.ndarray)
    assert_equal(my_method_S.s, [])
    assert_equal(my_method_S.a, [])
    assert_equal(my_method_S.alfa, 0.10)
    assert_equal(my_method_S.gamma, 0.90)
    assert_equal(my_method_S.epsilon, 0.10)
    assert_equal(my_method_S.episodes_max, 1)
    assert_equal(my_method_S.steps_max, 10)


@pytest.mark.parametrize('my_method_S', ['24', '20', '12'], indirect=True)
def test_choose_action_S(my_method_S):
    """Test that the function choose_action takes the row "0" of the

    Q array when p < 1 - epsilon or takes a random row otherwise.
    """
    method = my_method_S
    method.ini_saq()
    i = method.choose_action(np.random.randint(624))

    # assert (isinstance(i, int))
    assert_equal(i, 0)


@patch.object(sim.Sarsa, 'process', return_value=[1., 0., 0., 2., 2.])
def test_process_S(mock_method3):
    """Test that the process function returns an array."""
    r = sim.Sarsa.process()
    mock_method3.assert_called_with()

    assert isinstance(r, list)
