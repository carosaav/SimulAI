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


# ============================================================================
# TESTS
# ============================================================================


class Test_Com:
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

    @pytest.fixture
    def com(self):
        # Here we import interface with win32 patched
        import simulai

        return simulai.CommunicationInterface("MaterialHandling.spp")

    @pytest.fixture
    def espera(self):
        import simulai

        return simulai.DiscreteVariable(
            "Espera", 60, 300, 10, "Models.Modelo.espera"
        )

    @pytest.fixture
    def espera_stp60(self):
        import simulai

        return simulai.DiscreteVariable(
            "Espera", 60, 300, 60, "Models.Modelo.espera"
        )

    @pytest.fixture
    def stock(self):
        import simulai

        return simulai.DiscreteVariable(
            "Stock", 10, 50, 10, "Models.Modelo.stock"
        )

    @pytest.fixture
    def numviajes(self):
        import simulai

        return simulai.DiscreteVariable(
            "Numero de viajes", 1, 5, 1, "Models.Modelo.numviajes"
        )

    @pytest.fixture
    def transportes(self):
        import simulai

        return simulai.OutcomeVariable(
            "Distancia Transportes", "Models.Modelo.transportes", 2, 9
        )

    @pytest.fixture
    def buffers(self):
        import simulai

        return simulai.OutcomeVariable(
            "Llenado buffers", "Models.Modelo.buffers", 3, 20
        )

    @pytest.fixture
    def salidas(self):
        import simulai

        return simulai.OutcomeVariable(
            "Espera en las Salidas", "Models.Modelo.salidas", 2, 20
        )

    @pytest.fixture
    def my_method_q(self, espera, stock, numviajes):
        import simulai

        return simulai.Qlearning(
            v_i=[espera, stock, numviajes], episodes_max=1, steps_max=10
        )

    @pytest.fixture
    def my_method_s(self, espera, stock, numviajes):
        import simulai

        return simulai.Sarsa(
            v_i=[espera, stock, numviajes], episodes_max=1, steps_max=10
        )

    @pytest.fixture
    def base(
        self,
        espera,
        stock,
        numviajes,
        transportes,
        buffers,
        salidas,
        my_method_q,
    ):
        import simulai

        return simulai.BasePlant(
            method=my_method_q,
            v_i=[espera, stock, numviajes],
            v_o=[transportes, buffers, salidas],
            filename="MaterialHandling.spp",
            modelname="Model",
        )

    # =================================================
    # Now the actual testing
    # =================================================

    @patch("win32com.client.Dispatch")
    def test_process_ql(
        self,
        dispatch,
        com,
        espera,
        stock,
        numviajes,
        transportes,
        buffers,
        salidas,
    ):
        import simulai

        pcss = simulai.Qlearning(
            v_i=[
                simulai.DiscreteVariable(
                    "Espera", 60, 300, 10, "Models.Modelo.espera"
                ),
                simulai.DiscreteVariable(
                    "Stock", 10, 50, 10, "Models.Modelo.stock"
                ),
                simulai.DiscreteVariable(
                    "Numero de viajes", 1, 5, 1, "Models.Modelo.numviajes"
                ),
            ],
            episodes_max=1,
            steps_max=10,
        )
        b = simulai.BasePlant(
            method=pcss,
            v_i=[espera, stock, numviajes],
            v_o=[transportes, buffers, salidas],
            filename="MaterialHandling.spp",
            modelname="Model",
        )

        b.connection()

        r_episodes, s0 = pcss.process()

        assert isinstance(r_episodes, np.ndarray), "Should be an array"
        assert isinstance(s0, np.ndarray), "Should be an array"

    @patch("win32com.client.Dispatch")
    def test_process_sarsa(
        self,
        dispatch,
        com,
        espera,
        stock,
        numviajes,
        transportes,
        buffers,
        salidas,
    ):
        import simulai

        pcss = simulai.Sarsa(
            v_i=[
                simulai.DiscreteVariable(
                    "Espera", 60, 300, 10, "Models.Modelo.espera"
                ),
                simulai.DiscreteVariable(
                    "Stock", 10, 50, 10, "Models.Modelo.stock"
                ),
                simulai.DiscreteVariable(
                    "Numero de viajes", 1, 5, 1, "Models.Modelo.numviajes"
                ),
            ],
            episodes_max=1,
            steps_max=10,
            seed=24,
        )
        b = simulai.BasePlant(
            method=pcss,
            v_i=[espera, stock, numviajes],
            v_o=[transportes, buffers, salidas],
            filename="MaterialHandling.spp",
            modelname="Model",
        )

        b.connection()

        r_episodes, s0, a0 = pcss.process()

        assert isinstance(r_episodes, np.ndarray), "Should be an array"
        assert isinstance(s0, np.ndarray), "Should be an array"
        assert a0 == 0

    @patch("win32com.client.Dispatch")
    def test_discretevariable(self, dispatch, com, espera):
        import simulai

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
                "Espera", -60, 300, 10, "Models.Modelo.espera"
            )
        with pytest.raises(ValueError):
            simulai.DiscreteVariable(
                "Espera", 60, -300, 10, "Models.Modelo.espera"
            )
        with pytest.raises(ValueError):
            simulai.DiscreteVariable(
                "Espera", 60, 300, -10, "Models.Modelo.espera"
            )

    @patch("win32com.client.Dispatch")
    def test_outcomevariable(self, dispatch, com, transportes):
        import simulai

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
            simulai.OutcomeVariable("Distance", "Model", -2, 9)
        with pytest.raises(ValueError):
            simulai.OutcomeVariable("Distance", "Model", 2, -9)

    @patch("win32com.client.Dispatch")
    def test_baseplant(
        self,
        dispatch,
        com,
        base,
        my_method_q,
        espera,
        stock,
        numviajes,
        transportes,
        buffers,
        salidas,
    ):
        import simulai

        assert isinstance(base.v_i, list), "Should be a list"
        assert isinstance(base.v_o, list), "Should be a list"
        assert isinstance(base.filename, str), "Should be a string"
        assert isinstance(base.modelname, str), "Should be a string"

        with pytest.raises(TypeError):
            simulai.BasePlant(
                my_method_q,
                1,
                [transportes, buffers, salidas],
                "MH.spp",
                "frame",
            )
        with pytest.raises(TypeError):
            simulai.BasePlant(
                my_method_q, [espera, stock, numviajes], 2.0, "MH.spp", "frame"
            )
        with pytest.raises(TypeError):
            simulai.BasePlant(
                my_method_q,
                [espera, stock, numviajes],
                [transportes, buffers, salidas],
                10,
                "frame",
            )
        with pytest.raises(TypeError):
            simulai.BasePlant(
                my_method_q,
                [espera, stock, numviajes],
                [transportes, buffers, salidas],
                "MH.spp",
                10,
            )

    @patch("win32com.client.Dispatch")
    def test_get_file_name_plant(self, dispatch, com, base):
        filename = base.get_file_name_plant()

        assert filename == "MaterialHandling.spp"
        assert isinstance(filename, str), "Should be a string"

    @patch("win32com.client.Dispatch")
    def test_process_simulation(self, dispatch, com, base):
        base.process_simulation()
        dispatch.assert_called_once()

    @patch("win32com.client.Dispatch")
    def test_ini_saq1(self, dispatch, com, espera, stock, numviajes):
        import simulai

        basem = simulai.Qlearning(v_i=[espera], episodes_max=1, steps_max=10)
        basem.ini_saq()

        assert isinstance(basem.Q, np.ndarray)
        assert isinstance(basem.S, np.ndarray)
        assert isinstance(basem.actions, np.ndarray)
        assert basem.Q.shape == (25, 3)
        assert basem.S.shape == (25,)
        assert basem.actions.shape == (3,)
        assert (basem.Q == 0).all()
        assert bool((basem.S == 0).all()) is False
        assert bool((basem.actions == 0).all()) is False

        with pytest.raises(Exception):
            basen = simulai.Qlearning(
                v_i=[espera, stock, numviajes, espera, stock],
                episodes_max=1,
                steps_max=10,
            )
            basen.ini_saq()
        with pytest.raises(Exception):
            basel = simulai.Qlearning(
                v_i=[
                    simulai.DiscreteVariable(
                        "Espera", 10, 10000, 1, "Models.Modelo.espera"
                    )
                ],
                episodes_max=1,
                steps_max=10,
            )
            basel.ini_saq()

    @patch("win32com.client.Dispatch")
    def test_ini_saq2(self, dispatch, com, espera, stock, numviajes):
        import simulai

        basem = simulai.Qlearning(
            v_i=[espera, stock], episodes_max=1, steps_max=10
        )
        basem.ini_saq()

        assert isinstance(basem.Q, np.ndarray)
        assert isinstance(basem.S, np.ndarray)
        assert isinstance(basem.actions, np.ndarray)
        assert basem.Q.shape == (125, 9)
        assert basem.S.shape == (125, 2)
        assert basem.actions.shape == (9, 2)
        assert (basem.Q == 0).all()
        assert bool((basem.S == 0).all()) is False
        assert bool((basem.actions == 0).all()) is False

        with pytest.raises(Exception):
            basen = simulai.Qlearning(
                v_i=[espera, stock, numviajes, espera, stock],
                episodes_max=1,
                steps_max=10,
            )
            basen.ini_saq()
        with pytest.raises(Exception):
            basel = simulai.Qlearning(
                v_i=[
                    simulai.DiscreteVariable(
                        "Espera", 10, 10000, 1, "Models.Modelo.espera"
                    )
                ],
                episodes_max=1,
                steps_max=10,
            )
            basel.ini_saq()

    @patch("win32com.client.Dispatch")
    def test_ini_saq3(self, dispatch, com, espera, stock, numviajes):
        import simulai

        basem = simulai.Qlearning(
            v_i=[espera, stock, numviajes], episodes_max=1, steps_max=10
        )
        basem.ini_saq()

        assert isinstance(basem.Q, np.ndarray)
        assert isinstance(basem.S, np.ndarray)
        assert isinstance(basem.actions, np.ndarray)
        assert basem.Q.shape == (625, 27)
        assert basem.S.shape == (625, 3)
        assert basem.actions.shape == (27, 3)
        assert (basem.Q == 0).all()
        assert bool((basem.S == 0).all()) is False
        assert bool((basem.actions == 0).all()) is False

        with pytest.raises(Exception):
            basen = simulai.Qlearning(
                v_i=[espera, stock, numviajes, espera, stock],
                episodes_max=1,
                steps_max=10,
            )
            basen.ini_saq()
        with pytest.raises(Exception):
            basel = simulai.Qlearning(
                v_i=[
                    simulai.DiscreteVariable(
                        "Espera", 10, 10000, 1, "Models.Modelo.espera"
                    )
                ],
                episodes_max=1,
                steps_max=10,
            )
            basel.ini_saq()

    @patch("win32com.client.Dispatch")
    def test_ini_saq4(self, dispatch, com, espera_stp60, stock, numviajes):
        import simulai

        basem = simulai.Qlearning(
            v_i=[espera_stp60, stock, numviajes, espera_stp60],
            episodes_max=1,
            steps_max=10,
        )
        basem.ini_saq()

        assert isinstance(basem.Q, np.ndarray)
        assert isinstance(basem.S, np.ndarray)
        assert isinstance(basem.actions, np.ndarray)
        assert basem.Q.shape == (625, 81)
        assert basem.S.shape == (625, 4)
        assert basem.actions.shape == (81, 4)
        assert (basem.Q == 0).all()
        assert bool((basem.S == 0).all()) is False
        assert bool((basem.actions == 0).all()) is False

        with pytest.raises(Exception):
            basen = simulai.Qlearning(
                v_i=[espera_stp60, stock, numviajes, espera_stp60, stock],
                episodes_max=1,
                steps_max=10,
            )
            basen.ini_saq()
        with pytest.raises(Exception):
            basel = simulai.Qlearning(
                v_i=[
                    simulai.DiscreteVariable(
                        "Espera", 10, 10000, 1, "Models.Modelo.espera"
                    )
                ],
                episodes_max=1,
                steps_max=10,
            )
            basel.ini_saq()

    @patch("win32com.client.Dispatch")
    def test_qlearning(
        self, dispatch, com, my_method_q, espera, stock, numviajes
    ):
        import simulai

        ql = my_method_q

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
            simulai.Qlearning([espera, stock, numviajes], 3.0, 10)
        with pytest.raises(TypeError):
            simulai.Qlearning([espera, stock, numviajes], 10, "nine")
        with pytest.raises(TypeError):
            simulai.Qlearning([espera, stock, numviajes], 10, 10, alfa=2)
        with pytest.raises(TypeError):
            simulai.Qlearning([espera, stock, numviajes], 10, 10, gamma=2)
        with pytest.raises(TypeError):
            simulai.Qlearning([espera, stock, numviajes], 10, 10, epsilon=2)

        with pytest.raises(Exception):
            simulai.Qlearning(
                10,
                10,
                v_i=["espera", "stock", "numviajes", "tiempo", "velocidad"],
            )

        with pytest.raises(ValueError):
            simulai.Qlearning([espera, stock, numviajes], -10, 10)
        with pytest.raises(ValueError):
            simulai.Qlearning([espera, stock, numviajes], 10, -10)
        with pytest.raises(ValueError):
            simulai.Qlearning([espera, stock, numviajes], 10, 10, alfa=-2.0)
        with pytest.raises(ValueError):
            simulai.Qlearning([espera, stock, numviajes], 10, 10, alfa=2.0)
        with pytest.raises(ValueError):
            simulai.Qlearning([espera, stock, numviajes], 10, 10, gamma=-2.0)
        with pytest.raises(ValueError):
            simulai.Qlearning([espera, stock, numviajes], 10, 10, gamma=2.0)
        with pytest.raises(ValueError):
            simulai.Qlearning([espera, stock, numviajes], 10, 10, epsilon=-2.0)
        with pytest.raises(ValueError):
            simulai.Qlearning([espera, stock, numviajes], 10, 10, epsilon=2.0)

    @patch("win32com.client.Dispatch")
    def test_arrays(self, dispatch, com, espera, stock, numviajes):
        import simulai

        ql = simulai.Qlearning(
            v_i=[espera, stock, numviajes], episodes_max=1, steps_max=10
        )
        ql.arrays()
        assert len(ql.s) == 3
        assert len(ql.a) == 3

    @pytest.mark.parametrize(
        "seed_input, expected", [(24, 0), (20, 0), (12, 0)]
    )
    @patch("win32com.client.Dispatch")
    def test_choose_action(
        self, dispatch, com, espera, stock, numviajes, seed_input, expected
    ):
        import simulai

        method = simulai.Qlearning(
            v_i=[espera, stock, numviajes],
            episodes_max=1,
            steps_max=10,
            seed=seed_input,
        )
        method.ini_saq()
        i = method.choose_action(np.random.randint(624))
        assert i == expected

    @patch("win32com.client.Dispatch")
    def test_q_sarsa(self, dispatch, com, my_method_q, my_method_s):
        method1 = my_method_q
        method2 = my_method_s

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
