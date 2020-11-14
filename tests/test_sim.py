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
    espera = sim.DiscreteVariable("Espera", 60, 300, 10,
                                  "Models.Modelo.espera")
    return espera


@pytest.fixture
def stock():
    stock = sim.DiscreteVariable("Stock", 10, 50, 10, "Models.Modelo.stock")
    return stock


@pytest.fixture
def numviajes():
    numviajes = sim.DiscreteVariable(
        "Numero de viajes", 1, 5, 1, "Models.Modelo.numviajes"
    )
    return numviajes


@pytest.fixture
def transportes():
    transportes = sim.OutcomeVariable(
        "Distancia Transportes", "Models.Modelo.transportes", 2, 9
    )
    return transportes


@pytest.fixture
def buffers():
    buffers = sim.OutcomeVariable("Llenado buffers",
                                  "Models.Modelo.buffers", 3, 20)
    return buffers


@pytest.fixture
def salidas():
    salidas = sim.OutcomeVariable(
        "Espera en las Salidas", "Models.Modelo.salidas", 2, 20
    )
    return salidas


@pytest.fixture
def var_out(transportes, buffers, salidas):
    vo = [transportes, buffers, salidas]

    return vo


@pytest.fixture
def my_method_Q(var_input):
    method = sim.Qlearning(v_i=var_input, episodes_max=1,
                           steps_max=10, seed=None)

    return method


@pytest.fixture
def my_method_S(var_input):
    method = sim.Sarsa(v_i=var_input, episodes_max=1, steps_max=10, seed=None)

    return method


@pytest.fixture
def base(var_input, var_out, my_method_Q):
    plant = sim.BasePlant(
        method=my_method_Q,
        v_i=var_input,
        v_o=var_out,
        filename="MaterialHandling.spp",
        modelname="Model",
    )

    return plant


@pytest.mark.parametrize(
    "namef, lowf, upf, stf, pathf",
    [
        ("Espera", 60.0, 300, 10, "Models.Modelo.espera"),
        ("Espera", 60, 300.0, 10, "Models.Modelo.espera"),
        ("Espera", 60, 300, 10.0, "Models.Modelo.espera"),
        (["Espera"], 60, 300, 10, "Models.Modelo.espera"),
        ({"e": "Espera"}, 60, 300, 10, "Models.Modelo.espera"),
        ("Espera", 60, 300, (4.5 + 3j), "Models.Modelo.espera"),
        ("Espera", 60, 300, 10, False),
    ],
)
def test_DiscreteVariable(namef, lowf, upf, stf, pathf):
    parm = sim.DiscreteVariable("Espera", 60, 300, 10, "Models.Modelo.espera")

    assert isinstance(parm.name, str), "Should be a string"
    assert isinstance(parm.lower_limit, int), "Should be an integer"
    assert isinstance(parm.upper_limit, int), "Should be an integer"
    assert isinstance(parm.step, int), "Should be an integer"
    assert isinstance(parm.path, str), "Should be a string"

    with pytest.raises(TypeError):
        sim.DiscreteVariable(namef, lowf, upf, stf, pathf)

    with pytest.raises(ValueError):
        sim.DiscreteVariable("Espera", -60, 300, 10, "Models.Modelo.espera")
        sim.DiscreteVariable("Espera", 60, -300, 10, "Models.Modelo.espera")
        sim.DiscreteVariable("Espera", 60, 300, -10, "Models.Modelo.espera")


@pytest.mark.parametrize(
    "namef, pathf, colf, rowf",
    [
        (False, "Model", 2, 9),
        ("Distance", "Model", 2.0, 9),
        ("Distance", True, 2, 9.0),
        (4.2, "Model", 2, 9),
        ("Distance", {"m": "Model"}, 2, 9),
        ("Distance", "Model", 2, "nine"),
    ],
)
def test_OutcomeVariable(namef, pathf, colf, rowf):
    parm = sim.OutcomeVariable("Time", "path", 5, 1)

    assert isinstance(parm.name, str), "Should be a string"
    assert isinstance(parm.path, str), "Should be a string"
    assert isinstance(parm.column, int), "Should be a integer"
    assert isinstance(parm.num_rows, int), "Should be a integer"

    with pytest.raises(TypeError):
        sim.OutcomeVariable(namef, pathf, colf, rowf)

    with pytest.raises(ValueError):
        sim.OutcomeVariable("Time", "path", -5, 1)
        sim.OutcomeVariable("Time", "path", 5, -1)


def test_BasePlant(base, var_input, var_out):

    assert isinstance(base.v_i, list), "Should be a list"
    assert isinstance(base.v_o, list), "Should be a list"
    assert isinstance(base.filename, str), "Should be a string"
    assert isinstance(base.modelname, str), "Should be a string"

    with pytest.raises(TypeError):
        sim.BasePlant(1, var_out, "MH.spp", "frame")
        sim.BasePlant(var_input, 2., "MH.spp", "frame")
        sim.BasePlant(var_input, var_out, 10, "frame")
        sim.BasePlant(var_input, var_out, "MH.spp", 10)


def test_get_file_name_plant(base):
    filename = base.get_file_name_plant()

    assert filename == "MaterialHandling.spp"
    assert isinstance(filename, str), "Should be a string"


@patch.object(sim.BasePlant, "update", return_value=np.random.uniform(0, 5))
def test_update(update):
    value = sim.BasePlant.update([60, 10, 1])
    update.assert_called_with([60, 10, 1])

    assert isinstance(value, float), "Should be a float"


@pytest.mark.parametrize(
    "var_input, epmax, stmax", [([espera, stock, numviajes], 1, 10)]
)
@patch.multiple(sim.BaseMethod, __abstractmethods__=set())
def test_BaseMethod(var_input, epmax, stmax):
    BaseM = sim.BaseMethod(
        v_i=var_input, episodes_max=epmax, steps_max=stmax, seed=None
    )

    assert isinstance(BaseM.s, list), "Should be a list"
    assert isinstance(BaseM.a, list), "Should be a list"
    assert isinstance(BaseM.v_i, list), "Should be a list"
    assert isinstance(BaseM.alfa, float), "Should be a float"
    assert isinstance(BaseM.gamma, float), "Should be a float"
    assert isinstance(BaseM.epsilon, float), "Should be a float"
    assert isinstance(BaseM.episodes_max, int), "Should be an integer"
    assert isinstance(BaseM.steps_max, int), "Should be an integer"
    assert isinstance(BaseM.r_episode, np.ndarray), "Should be an array"
    assert_equal(len(BaseM.s), 0)
    assert_equal(len(BaseM.a), 0)
    assert_equal(BaseM.alfa, 0.10)
    assert_equal(BaseM.gamma, 0.90)
    assert_equal(BaseM.epsilon, 0.10)
    assert_equal(BaseM.episodes_max, 1)
    assert_equal(BaseM.steps_max, 10)

    with pytest.raises(TypeError):
        sim.BaseMethod("variable", epmax, stmax)
        sim.BaseMethod(var_input, 3., stmax)
        sim.BaseMethod(var_input, epmax, "nine")
        sim.BaseMethod(var_input, epmax, stmax, alfa="zero")
        sim.BaseMethod(var_input, epmax, stmax, gamma=2)
        sim.BaseMethod(var_input, epmax, epsilon=1)

    with pytest.raises(Exception):
        sim.BaseMethod(epmax, stmax, v_i=["espera", "stock", "numviajes",
                       "tiempo", "velocidad"])

    with pytest.raises(ValueError):
        sim.BaseMethod(var_input, -1, stmax)
        sim.BaseMethod(var_input, epmax, -1)
        sim.BaseMethod(var_input, epmax, stmax, alfa=-1)
        sim.BaseMethod(var_input, epmax, stmax, alfa=3)
        sim.BaseMethod(var_input, epmax, stmax, gamma=-1)
        sim.BaseMethod(var_input, epmax, stmax, gamma=3)
        sim.BaseMethod(var_input, epmax, stmax, epsilon=-1)
        sim.BaseMethod(var_input, epmax, stmax, epsilon=3)


@pytest.mark.xfail
@patch.multiple(sim.BaseMethod, __abstractmethods__=set())
def test_ini_saq(var_input):
    baseM = sim.BaseMethod(v_i=var_input, episodes_max=1, steps_max=10)
    initial = baseM.ini_saq()

    assert isinstance(initial.n, list), "Should be a list"
    assert isinstance(initial.m, list), "Should be a list"
    assert isinstance(initial.Q, np.ndarray)
    assert isinstance(initial.S, np.ndarray)
    assert isinstance(initial.actions, np.ndarray)
    assert initial.Q.shape == (625, 27)
    assert initial.S.shape == (625, 3)
    assert initial.actions.shape == (27, 3)
    assert (initial.Q == 0).all()
    assert bool((initial.S == 0).all()) is False
    assert bool((initial.actions == 0).all()) is False


@pytest.mark.parametrize("seed_input, expected", [(24, 0), (20, 0), (12, 0)])
def test_choose_action_Q(var_input, seed_input, expected):
    method = sim.Qlearning(v_i=var_input, episodes_max=1, steps_max=10,
                           seed=seed_input)
    method.ini_saq()
    i = method.choose_action(np.random.randint(624))

    assert_equal(i, expected)


@pytest.mark.xfail
def test_process_Q(my_method_Q):
    process = my_method_Q.process()

    assert isinstance(process.r_episode, list), "Should be a list"
    assert isinstance(process.S0, float), "Should be a float"
    assert isinstance(process.t, int), "Should be a int"
    assert isinstance(process.r_acum, float), "Should be a float"
    assert isinstance(process.res0, float), "Should be a float"
    assert isinstance(process.j, int), "Should be a int"
    assert isinstance(process.Snew, float), "Should be a float"
    assert isinstance(process.res1, float), "Should be a float"
    assert isinstance(process.r, int), "Should be a int"


@pytest.mark.parametrize("seed_input, expected", [(24, 0), (20, 0), (12, 0)])
def test_choose_action_S(var_input, seed_input, expected):
    method = sim.Sarsa(v_i=var_input, episodes_max=1, steps_max=10,
                       seed=seed_input)
    method.ini_saq()
    i = method.choose_action(np.random.randint(624))

    assert_equal(i, expected)


@pytest.mark.xfail
def test_process_S(my_method_S):
    process = my_method_S.process()

    assert isinstance(process.r_episode, list), "Should be a list"
    assert isinstance(process.S0, float), "Should be a float"
    assert isinstance(process.A0, int), "Should be a int"
    assert isinstance(process.t, int), "Should be a float"
    assert isinstance(process.r_acum, float), "Should be a float"
    assert isinstance(process.res0, float), "Should be a float"
    assert isinstance(process.j, int), "Should be a int"
    assert isinstance(process.Snew, float), "Should be a float"
    assert isinstance(process.Anew, int), "Should be a int"
    assert isinstance(process.res1, float), "Should be a float"
    assert isinstance(process.r, int), "Should be a int"


def test_Q_SARSA(my_method_Q, my_method_S):
    method1 = my_method_Q
    method2 = my_method_S

    assert_equal(method1.v_i, method2.v_i)
    assert_equal(method1.episodes_max, method2.episodes_max)
    assert_equal(method1.steps_max, method2.steps_max)
    assert_equal(method1.s, method2.s)
    assert_equal(method1.a, method2.a)
    assert_equal(method1.s, method2.s)
    assert_equal(method1.alfa, method2.alfa)
    assert_equal(method1.gamma, method2.gamma)
    assert_equal(method1.epsilon, method2.epsilon)
