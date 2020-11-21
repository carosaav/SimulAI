
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
from numpy.testing import assert_equal
import numpy as np


# ============================================================================
# TESTS
# ============================================================================


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
            "win32com.client.Dispatch": self.win32com.client.Dispatch
            }

        self.module_patcher = patch.dict("sys.modules", mock_modules)
        self.module_patcher.start()

    def teardown_method(self):
        self.module_patcher.stop()

    @pytest.fixture
    def com(self):
        # Here we import interface with win32 patched
        import simulai
        return simulai.CommunicationInterface('MaterialHandling.spp')

    @pytest.fixture
    def espera(self):
        import simulai
        return simulai.DiscreteVariable("Espera", 60, 300, 10,
                                        "Models.Modelo.espera")

    @pytest.fixture
    def espera_mal(self):
        import simulai
        return simulai.DiscreteVariable("Espera", -60, -300, -10,
                                        "Models.Modelo.espera")

    @pytest.fixture
    def stock(self):
        import simulai
        return simulai.DiscreteVariable(
            "Stock", 10, 50, 10, "Models.Modelo.stock")

    @pytest.fixture
    def numviajes(self):
        import simulai
        return simulai.DiscreteVariable(
            "Numero de viajes", 1, 5, 1, "Models.Modelo.numviajes")

    @pytest.fixture
    def transportes(self):
        import simulai
        return simulai.OutcomeVariable(
            "Distancia Transportes", "Models.Modelo.transportes", 2, 9)

    @pytest.fixture
    def transportes_mal(self):
        import simulai
        return simulai.OutcomeVariable(
            "Distancia Transportes", "Models.Modelo.transportes", -2, -9)

    @pytest.fixture
    def buffers(self):
        import simulai
        return simulai.OutcomeVariable("Llenado buffers",
                                       "Models.Modelo.buffers", 3, 20)

    @pytest.fixture
    def salidas(self):
        import simulai
        return simulai.OutcomeVariable(
            "Espera en las Salidas", "Models.Modelo.salidas", 2, 20)

    @pytest.fixture
    def var_input(espera, stock, numviajes):
        vi = [espera, stock, numviajes]
        return vi

    @pytest.fixture
    def var_out(transportes, buffers, salidas):
        vo = [transportes, buffers, salidas]
        return vo

    @pytest.fixture
    def my_method_Q(self, var_input):
        import simulai
        return simulai.Qlearning(v_i=var_input,
                                 episodes_max=1, steps_max=10)

    @pytest.fixture
    def my_method_Q_mal(self, var_input):
        import simulai
        return simulai.Qlearning(v_i=var_input, episodes_max=-1,
                                 steps_max=-10, alfa=-1, gamma=-1, epsilon=-1)

    @pytest.fixture
    def my_method_Q_mal2(self, var_input):
        import simulai
        return simulai.Qlearning(v_i=var_input, episodes_max=1,
                                 steps_max=10, alfa=3, gamma=3, epsilon=3)

    @pytest.fixture
    def my_method_S(self, var_input):
        import simulai
        return simulai.Sarsa(v_i=var_input, episodes_max=1, steps_max=10)

    @pytest.fixture
    def base(self, var_input, var_out, my_method_Q):
        import simulai
        return simulai.BasePlant(method=my_method_Q, v_i=var_input,
                                 v_o=var_out, filename="MaterialHandling.spp",
                                 modelname="Model")

    # =================================================
    # Now the actual testing
    # =================================================

    # @pytest.mark.parametrize(
    #     "namef, lowf, upf, stf, pathf",
    #     [
    #         ("Espera", 60.0, 300, 10, "Models.Modelo.espera"),
    #         ("Espera", 60, 300.0, 10, "Models.Modelo.espera"),
    #         ("Espera", 60, 300, 10.0, "Models.Modelo.espera"),
    #         (["Espera"], 60, 300, 10, "Models.Modelo.espera"),
    #         ({"e": "Espera"}, 60, 300, 10, "Models.Modelo.espera"),
    #         ("Espera", 60, 300, (4.5 + 3j), "Models.Modelo.espera"),
    #         ("Espera", 60, 300, 10, False),
    #     ],
    # )
    def test_DiscreteVariable(self, espera, espera_mal):
        parm = espera
        assert isinstance(parm.name, str), "Should be a string"
        assert isinstance(parm.lower_limit, int), "Should be an integer"
        assert isinstance(parm.upper_limit, int), "Should be an integer"
        assert isinstance(parm.step, int), "Should be an integer"
        assert isinstance(parm.path, str), "Should be a string"

        # with pytest.raises(TypeError):
        #     simulai.DiscreteVariable(namef, lowf, upf, stf, pathf)

        with pytest.raises(ValueError):
            espera_mal

    # @pytest.mark.parametrize(
    #     "namef, pathf, colf, rowf",
    #     [
    #         (False, "Model", 2, 9),
    #         ("Distance", "Model", 2.0, 9),
    #         ("Distance", True, 2, 9.0),
    #         (4.2, "Model", 2, 9),
    #         ("Distance", {"m": "Model"}, 2, 9),
    #         ("Distance", "Model", 2, "nine"),
    #     ],
    # )
    def test_OutcomeVariable(self, transportes, transportes_mal):
        parm = transportes

        assert isinstance(parm.name, str), "Should be a string"
        assert isinstance(parm.path, str), "Should be a string"
        assert isinstance(parm.column, int), "Should be a integer"
        assert isinstance(parm.num_rows, int), "Should be a integer"

        # with pytest.raises(TypeError):
        #     simulai.OutcomeVariable(namef, pathf, colf, rowf)

        with pytest.raises(ValueError):
            transportes_mal

    @patch('win32com.client.Dispatch')
    def test_BasePlant(self, dispatch, com, base):

        assert isinstance(base.v_i, list), "Should be a list"
        assert isinstance(base.v_o, list), "Should be a list"
        assert isinstance(base.filename, str), "Should be a string"
        assert isinstance(base.modelname, str), "Should be a string"

        # with pytest.raises(TypeError):
        #     simulai.BasePlant(my_method_Q, 1, var_out, "MH.spp", "frame")
        # with pytest.raises(TypeError):
        #     simulai.BasePlant(my_method_Q, var_input, 2.0, "MH.spp", "frame")
        # with pytest.raises(TypeError):
        #     simulai.BasePlant(my_method_Q, var_input, var_out, 10, "frame")
        # with pytest.raises(TypeError):
        #     simulai.BasePlant(my_method_Q, var_input, var_out, "MH.spp", 10)

    @patch('win32com.client.Dispatch')
    def test_get_file_name_plant(self, dispatch, com, base):
        filename = base.get_file_name_plant()

        assert filename == "MaterialHandling.spp"
        assert isinstance(filename, str), "Should be a string"

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_update(self, dispatch, com, base):
        value = base.update([60, 10, 1])
        base.update.assert_called_with([60, 10, 1])

        assert isinstance(value, float), "Should be a float"

    @patch('win32com.client.Dispatch')
    def test_Qlearning(self, my_method_Q, my_method_Q_mal, my_method_Q_mal2):
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
        assert_equal(len(ql.s), 0)
        assert_equal(len(ql.a), 0)
        assert_equal(ql.alfa, 0.10)
        assert_equal(ql.gamma, 0.90)
        assert_equal(ql.epsilon, 0.10)
        assert_equal(ql.episodes_max, 10)
        assert_equal(ql.steps_max, 10)

        # with pytest.raises(TypeError):
        #     simulai.Qlearning("variable", 10, 10)
        # with pytest.raises(TypeError):
        #     simulai.Qlearning(var_input, 3.0, 10)
        # with pytest.raises(TypeError):
        #     simulai.Qlearning(var_input, 10, "nine")
        # with pytest.raises(TypeError):
        #     simulai.Qlearning(var_input, 10, 10, alfa=2)
        # with pytest.raises(TypeError):
        #     simulai.Qlearning(var_input, 10, 10, gamma=2)
        # with pytest.raises(TypeError):
        #     simulai.Qlearning(var_input, 10, 10, epsilon=2)

        # with pytest.raises(Exception):
        #     simulai.Qlearning(
        #         10, 10, v_i=["espera", "stock", "numviajes",
        #                      "tiempo", "velocidad"]
        #     )

        with pytest.raises(ValueError):
            my_method_Q_mal
        with pytest.raises(ValueError):
            my_method_Q_mal2

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_arrays(self, dispatch, com, my_method_Q):
        q = my_method_Q
        q.arrays()
        assert_equal(len(q.s), 3)
        assert_equal(len(q.a), 3)

    # @pytest.mark.parametrize(
    #     "var_input, expQ, expS, expA",
    #     [
    #         (
    #             [simulai.DiscreteVariable("Espera", 60, 300, 10,
    #                                       "Models.Modelo.espera")],
    #             (25, 3),
    #             (25,),
    #             (3,),
    #         ),
    #         (
    #             [
    #                 simulai.DiscreteVariable("Espera", 60, 300, 10,
    #                                          "Models.Modelo.espera"),
    #                 simulai.DiscreteVariable("Stock", 10, 50, 10,
    #                                          "Models.Modelo.stock"),
    #             ],
    #             (125, 9),
    #             (125, 2),
    #             (9, 2),
    #         ),
    #         (
    #             [
    #                 simulai.DiscreteVariable("Espera", 60, 300, 10,
    #                                          "Models.Modelo.espera"),
    #                 simulai.DiscreteVariable("Stock", 10, 50, 10,
    #                                          "Models.Modelo.stock"),
    #                 simulai.DiscreteVariable("Numero de viajes", 1, 5, 1,
    #                                          "Models.Modelo.numviajes"),
    #             ],
    #             (625, 27),
    #             (625, 3),
    #             (27, 3),
    #         ),
    #         (
    #             [
    #                 simulai.DiscreteVariable("Espera", 60, 300, 60,
    #                                          "Models.Modelo.espera"),
    #                 simulai.DiscreteVariable("Stock", 10, 50, 10,
    #                                          "Models.Modelo.stock"),
    #                 simulai.DiscreteVariable("Numero de viajes", 1, 5, 1,
    #                                          "Models.Modelo.numviajes"),
    #                 simulai.DiscreteVariable("Espera", 60, 300, 60,
    #                                          "Models.Modelo.espera"),
    #             ],
    #             (625, 81),
    #             (625, 4),
    #             (81, 4),
    #         ),
    #     ],
    # )
    @patch('win32com.client.Dispatch')
    def test_ini_saq(self, var_input, expQ, expS, expA, my_method_Q):
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

        # with pytest.raises(Exception):
        #     baseN = simulai.Qlearning(
        #         v_i=[
        #             simulai.DiscreteVariable("Espera", 60, 300, 10,
        #                                      "Models.Modelo.espera"),
        #             simulai.DiscreteVariable("Stock", 10, 50, 10,
        #                                      "Models.Modelo.stock"),
        #             simulai.DiscreteVariable("Numero de viajes", 1, 5, 1,
        #                                      "Models.Modelo.numviajes"),
        #             simulai.DiscreteVariable("Espera", 60, 300, 10,
        #                                      "Models.Modelo.espera"),
        #             simulai.DiscreteVariable("Stock", 10, 50, 10,
        #                                      "Models.Modelo.stock"),
        #         ],
        #         episodes_max=1,
        #         steps_max=10,
        #     )
        #     baseN.ini_saq()
        # with pytest.raises(Exception):
        #     baseL = simulai.Qlearning(
        #         v_i=[
        #             simulai.DiscreteVariable("Espera", 10, 10000, 1,
        #                                      "Models.Modelo.espera")],
        #         episodes_max=1,
        #         steps_max=10,
        #     )
        #     baseL.ini_saq()

    # @pytest.mark.parametrize("seed_input, expected",
    #                          [(24, 0), (20, 0), (12, 0)])
    # def test_choose_action(var_input, seed_input, expected):
    #     method = simulai.Qlearning(v_i=var_input, episodes_max=1,
    #                                steps_max=10, seed=seed_input)
    #     method.ini_saq()
    #     i = method.choose_action(np.random.randint(624))

    #     assert_equal(i, expected)

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_process(self, dispatch, com, my_method_Q):
        value = my_method_Q.process()

        assert isinstance(value, float), "Should be a float"

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_process_S(self, dispatch, com, my_method_S):
        value = my_method_S.process()
        my_method_S.assert_called_with()

        assert isinstance(value, float), "Should be a float"

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_Q_SARSA(self, my_method_Q, my_method_S):
        method1 = my_method_Q
        method2 = my_method_S

        assert_equal(method1.v_i, method2.v_i)
        assert_equal(method1.episodes_max, method2.episodes_max)
        assert_equal(method1.steps_max, method2.steps_max)
        assert_equal(method1.s, method2.s)
        assert_equal(method1.a, method2.a)
        assert_equal(method1.seed, method2.seed)
        assert_equal(method1.alfa, method2.alfa)
        assert_equal(method1.gamma, method2.gamma)
        assert_equal(method1.epsilon, method2.epsilon)
        assert_equal(method1.r_episode, method2.r_episode)
