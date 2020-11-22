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
from unittest.mock import patch, MagicMock
import numpy as np
import simulai


# ============================================================================
# TESTS
# ============================================================================

@pytest.fixture
def espera():
    return simulai.DiscreteVariable(
        "Espera", 60, 300, 10, "Models.Modelo.espera"
    )

@pytest.fixture
def stock():
    return simulai.DiscreteVariable(
        "Stock", 10, 50, 10, "Models.Modelo.stock"
    )

@pytest.fixture
def numviajes():
    return simulai.DiscreteVariable(
        "Numero de viajes", 1, 5, 1, "Models.Modelo.numviajes"
    )

@pytest.fixture
def transportes():
    return simulai.OutcomeVariable(
        "Distancia Transportes", "Models.Modelo.transportes", 2, 9
    )

@pytest.fixture
def buffers():
    return simulai.OutcomeVariable(
        "Llenado buffers", "Models.Modelo.buffers", 3, 20
    )

@pytest.fixture
def salidas():
    return simulai.OutcomeVariable(
        "Espera en las Salidas", "Models.Modelo.salidas", 2, 20
    )

@pytest.fixture
def var_input(espera, stock, numviajes):
    vi = [espera, stock, numviajes]
    return vi

@pytest.fixture
def var_out(transportes, buffers, salidas):
    vo = [transportes, buffers, salidas]
    return vo

@pytest.fixture
def my_method_Q(var_input):
    return simulai.Qlearning(v_i=var_input, episodes_max=1, steps_max=10)

@pytest.fixture
def my_method_S(var_input):
    return simulai.Sarsa(v_i=var_input, episodes_max=1, steps_max=10)

@pytest.fixture
def base(var_input, var_out, my_method_Q):
    return simulai.BasePlant(
        method=my_method_Q,
        v_i=var_input,
        v_o=var_out,
        filename="MaterialHandling.spp",
        modelname="Model",
    )

class Test_Sim:
    # =================================================
    # General setup to mock an external api
    # This setup/teardown will be executed for each test
    # =================================================
    def setup_method(self):
        self.win32com = MagicMock()
        self.win32com.client = MagicMock()
        self.win32com.client.Dispatch = MagicMock()
        mock_modules = {
            "win32com": self.win32com,
            "win32com.client": self.win32com.client,
            "win32com.client.Dispatch": self.win32com.client.Dispatch,
        }

        self.module_patcher = patch.dict("sys.modules", mock_modules)
        self.module_patcher.start()

    def teardown_method(self):
        self.module_patcher.stop()

    # =================================================
    # Now the actual testing
    # =================================================

    @pytest.mark.xfail
    @patch("win32com.client.Dispatch")
    def test_update(self, dispatch, com, base, my_method_Q):
        value = base.update([60, 10, 1])
        base.update.assert_called_with([60, 10, 1])

        assert isinstance(value, float), "Should be a float"

    def test_process_Ql(self, var_input, var_out):
        pcss = simulai.Qlearning(v_i=[
                                 simulai.DiscreteVariable(
                                 "Espera", 60, 300, 10, "Models.Modelo.espera"
                                 ), simulai.DiscreteVariable(
                                 "Stock", 10, 50, 10, "Models.Modelo.stock"
                                 ), simulai.DiscreteVariable(
                                 "Numero de viajes", 1, 5, 1,
                                 "Models.Modelo.numviajes")],
                                 episodes_max=1, steps_max=10)
        b = simulai.BasePlant(
            method=pcss,
            v_i=var_input,
            v_o=var_out,
            filename="MaterialHandling.spp",
            modelname="Model",
        )

        b.connection()

        r_episodes, s0 = pcss.process()

        assert isinstance(r_episodes, np.ndarray), "Should be an array"
        assert isinstance(s0, np.ndarray), "Should be an array"

    def test_process_Sarsa(self, var_input, var_out):
        pcss = simulai.Sarsa(v_i=[
                                 simulai.DiscreteVariable(
                                 "Espera", 60, 300, 10, "Models.Modelo.espera"
                                 ), simulai.DiscreteVariable(
                                 "Stock", 10, 50, 10, "Models.Modelo.stock"
                                 ), simulai.DiscreteVariable(
                                 "Numero de viajes", 1, 5, 1,
                                 "Models.Modelo.numviajes")],
                                 episodes_max=1, steps_max=10, seed=24)
        b = simulai.BasePlant(
            method=pcss,
            v_i=var_input,
            v_o=var_out,
            filename="MaterialHandling.spp",
            modelname="Model",
        )

        b.connection()

        r_episodes, s0, a0 = pcss.process()

        assert isinstance(r_episodes, np.ndarray), "Should be an array"
        assert isinstance(s0, np.ndarray), "Should be an array"
        assert a0 == 0


def test_DiscreteVariable(espera):
    parm = espera
    assert isinstance(parm.name, str), "Should be a string"
    assert isinstance(parm.lower_limit, int), "Should be an integer"
    assert isinstance(parm.upper_limit, int), "Should be an integer"
    assert isinstance(parm.step, int), "Should be an integer"
    assert isinstance(parm.path, str), "Should be a string"

    with pytest.raises(TypeError):
        simulai.DiscreteVariable(
            {"e": "Espera"}, 60, 300, 10, "Models.Modelo.espera"
        )
    with pytest.raises(TypeError):
        simulai.DiscreteVariable(
            "Espera", 60.0, 300, 10, "Models.Modelo.espera"
        )
    with pytest.raises(TypeError):
        simulai.DiscreteVariable(
            "Espera", 60, 300.0, 10, "Models.Modelo.espera"
        )
    with pytest.raises(TypeError):
        simulai.DiscreteVariable(
            "Espera", 60, 300, 10.0, "Models.Modelo.espera"
        )
    with pytest.raises(TypeError):
        simulai.DiscreteVariable("Espera", 60, 300, 10, False)

    with pytest.raises(ValueError):
        simulai.DiscreteVariable(
            "Espera", -60, -300, -10, "Models.Modelo.espera"
        )


def test_OutcomeVariable(transportes):
    parm = transportes

    assert isinstance(parm.name, str), "Should be a string"
    assert isinstance(parm.path, str), "Should be a string"
    assert isinstance(parm.column, int), "Should be a integer"
    assert isinstance(parm.num_rows, int), "Should be a integer"

    with pytest.raises(TypeError):
        simulai.OutcomeVariable("Distance", "Model", 2.0, 9)
    with pytest.raises(TypeError):
        simulai.OutcomeVariable("Distance", "Model", 2, 9.0)
    with pytest.raises(TypeError):
        simulai.OutcomeVariable("Distance", {"m": "Model"}, 2, 9)
    with pytest.raises(TypeError):
        simulai.OutcomeVariable(1, "Model", 2, 9)

    with pytest.raises(ValueError):
        simulai.OutcomeVariable("Distance", "Model", -2, -9)


def test_BasePlant(base, my_method_Q, var_input, var_out):

    assert isinstance(base.v_i, list), "Should be a list"
    assert isinstance(base.v_o, list), "Should be a list"
    assert isinstance(base.filename, str), "Should be a string"
    assert isinstance(base.modelname, str), "Should be a string"

    with pytest.raises(TypeError):
        simulai.BasePlant(my_method_Q, 1, var_out, "MH.spp", "frame")
    with pytest.raises(TypeError):
        simulai.BasePlant(my_method_Q, var_input, 2.0, "MH.spp", "frame")
    with pytest.raises(TypeError):
        simulai.BasePlant(my_method_Q, var_input, var_out, 10, "frame")
    with pytest.raises(TypeError):
        simulai.BasePlant(my_method_Q, var_input, var_out, "MH.spp", 10)


def test_get_file_name_plant(base):
    filename = base.get_file_name_plant()

    assert filename == "MaterialHandling.spp"
    assert isinstance(filename, str), "Should be a string"


def test_Qlearning(my_method_Q, var_input):
    ql = my_method_Q

    assert isinstance(ql.s, list), "Should be a list"
    assert isinstance(ql.a, list), "Should be a list"
    assert isinstance(ql.v_i, list), "Should be a list"
    assert isinstance(ql.alfa, float), "Should be a float"
    assert isinstance(ql.gamma, float), "Should be a float"
    assert isinstance(ql.epsilon, float), "Should be a float"
    assert isinstance(ql.episodes_max, int), "Should be an integer"
    assert isinstance(ql.steps_max, int), "Should be an integer"
    assert isinstance(ql.r_episode, np.ndarray), "Should be an array"
    assert len(ql.s) == 0
    assert len(ql.a) == 0
    assert ql.alfa == 0.10
    assert ql.gamma == 0.90
    assert ql.epsilon == 0.10
    assert ql.episodes_max == 1
    assert ql.steps_max == 10

    with pytest.raises(TypeError):
        simulai.Qlearning("variable", 10, 10)
    with pytest.raises(TypeError):
        simulai.Qlearning(var_input, 3.0, 10)
    with pytest.raises(TypeError):
        simulai.Qlearning(var_input, 10, "nine")
    with pytest.raises(TypeError):
        simulai.Qlearning(var_input, 10, 10, alfa=2)
    with pytest.raises(TypeError):
        simulai.Qlearning(var_input, 10, 10, gamma=2)
    with pytest.raises(TypeError):
        simulai.Qlearning(var_input, 10, 10, epsilon=2)

    with pytest.raises(Exception):
        simulai.Qlearning(
            10, 10, v_i=["espera", "stock", "numviajes",
                         "tiempo", "velocidad"]
        )

    with pytest.raises(ValueError):
        simulai.Qlearning(var_input, -10, 10)
    with pytest.raises(ValueError):
        simulai.Qlearning(var_input, 10, -10)
    with pytest.raises(ValueError):
        simulai.Qlearning(var_input, 10, 10, alfa=-2.)
    with pytest.raises(ValueError):
        simulai.Qlearning(var_input, 10, 10, alfa=2.)
    with pytest.raises(ValueError):
        simulai.Qlearning(var_input, 10, 10, gamma=-2.)
    with pytest.raises(ValueError):
        simulai.Qlearning(var_input, 10, 10, gamma=2.)
    with pytest.raises(ValueError):
        simulai.Qlearning(var_input, 10, 10, epsilon=-2.)
    with pytest.raises(ValueError):
        simulai.Qlearning(var_input, 10, 10, epsilon=2.)


def test_arrays(espera, stock, numviajes):
    ql = simulai.Qlearning(v_i=[espera, stock, numviajes],
                          episodes_max=1, steps_max=10)
    ql.arrays()
    assert len(ql.s) == 3
    assert len(ql.a) == 3


@pytest.mark.parametrize(
    "var_input, expQ, expS, expA",
    [
        (
            [simulai.DiscreteVariable("Espera", 60, 300, 10,
                                      "Models.Modelo.espera")],
            (25, 3),
            (25,),
            (3,),
        ),
        (
            [
                simulai.DiscreteVariable("Espera", 60, 300, 10,
                                         "Models.Modelo.espera"),
                simulai.DiscreteVariable("Stock", 10, 50, 10,
                                         "Models.Modelo.stock"),
            ],
            (125, 9),
            (125, 2),
            (9, 2),
        ),
        (
            [
                simulai.DiscreteVariable("Espera", 60, 300, 10,
                                         "Models.Modelo.espera"),
                simulai.DiscreteVariable("Stock", 10, 50, 10,
                                         "Models.Modelo.stock"),
                simulai.DiscreteVariable("Numero de viajes", 1, 5, 1,
                                         "Models.Modelo.numviajes"),
            ],
            (625, 27),
            (625, 3),
            (27, 3),
        ),
        (
            [
                simulai.DiscreteVariable("Espera", 60, 300, 60,
                                         "Models.Modelo.espera"),
                simulai.DiscreteVariable("Stock", 10, 50, 10,
                                         "Models.Modelo.stock"),
                simulai.DiscreteVariable("Numero de viajes", 1, 5, 1,
                                         "Models.Modelo.numviajes"),
                simulai.DiscreteVariable("Espera", 60, 300, 60,
                                         "Models.Modelo.espera"),
            ],
            (625, 81),
            (625, 4),
            (81, 4),
        ),
    ],
)
def test_ini_saq(expQ, expS, expA, my_method_Q):
    baseM = my_method_Q
    baseM.ini_saq()

    assert isinstance(baseM.Q, np.ndarray)
    assert isinstance(baseM.S, np.ndarray)
    assert isinstance(baseM.actions, np.ndarray)
    assert baseM.Q.shape == expQ
    assert baseM.S.shape == expS
    assert baseM.actions.shape == expA
    assert (baseM.Q == 0).all()
    assert bool((baseM.S == 0).all()) is False
    assert bool((baseM.actions == 0).all()) is False

    with pytest.raises(Exception):
        baseN = simulai.Qlearning(
            v_i=[
                simulai.DiscreteVariable("Espera", 60, 300, 10,
                                         "Models.Modelo.espera"),
                simulai.DiscreteVariable("Stock", 10, 50, 10,
                                         "Models.Modelo.stock"),
                simulai.DiscreteVariable("Numero de viajes", 1, 5, 1,
                                         "Models.Modelo.numviajes"),
                simulai.DiscreteVariable("Espera", 60, 300, 10,
                                         "Models.Modelo.espera"),
                simulai.DiscreteVariable("Stock", 10, 50, 10,
                                         "Models.Modelo.stock"),
            ],
            episodes_max=1,
            steps_max=10,
        )
        baseN.ini_saq()
    with pytest.raises(Exception):
        baseL = simulai.Qlearning(
            v_i=[
                simulai.DiscreteVariable("Espera", 10, 10000, 1,
                                         "Models.Modelo.espera")],
            episodes_max=1,
            steps_max=10,
        )
        baseL.ini_saq()
   

@pytest.mark.parametrize("seed_input, expected", [(24, 0), (20, 0), (12, 0)])
def test_choose_action(var_input, seed_input, expected):
    method = simulai.Qlearning(v_i=var_input, episodes_max=1, steps_max=10,
                           seed=seed_input)
    method.ini_saq()
    i = method.choose_action(np.random.randint(624))
    assert i == expected


def test_Q_SARSA(my_method_Q, my_method_S):
    method1 = my_method_Q
    method2 = my_method_S

    assert method1.v_i == method2.v_i
    assert method1.episodes_max == method2.episodes_max
    assert method1.steps_max == method2.steps_max
    assert method1.s == method2.s
    assert method1.a == method2.a
    assert method1.seed == method2.seed
    assert method1.alfa == method2.alfa
    assert method1.gamma == method2.gamma
    assert method1.epsilon == method2.epsilon
    assert method1.r_episode == method2.r_episode
