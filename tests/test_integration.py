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

    @patch("win32com.client.Dispatch")
    def test_example_1(self, dispatch):
        import simulai

        # Step 0 - Definition of model frame

        frame = "Modelo"

        # Step 1- Definition of input variables

        espera = simulai.DiscreteVariable(
            "Espera", 60, 300, 10, "Models."+frame+".espera")
        stock = simulai.DiscreteVariable(
            "Stock", 10, 50, 10, "Models."+frame+".stock")
        numviajes = simulai.DiscreteVariable(
            "Numero de viajes", 1, 5, 1, "Models."+frame+".numviajes")

        var_input = [espera, stock, numviajes]

        # Step 2- Definition of output variables

        transportes = simulai.OutcomeVariable(
            "Distancia Transportes", "Models."+frame+".transportes", 2, 9)
        buffers = simulai.OutcomeVariable(
            "Llenado buffers", "Models."+frame+".buffers", 3, 20)
        salidas = simulai.OutcomeVariable(
            "Espera en las Salidas", "Models."+frame+".salidas", 2, 20)

        var_output = [transportes, buffers, salidas]

        # Step 3- Choice of method, plant and simulation file.
        # Enter input variables and output variables.

        my_method = simulai.Qlearning(v_i=var_input, episodes_max=5, steps_max=10)
        my_plant = simulai.BasePlant(method=my_method, v_i=var_input, v_o=var_output,
                                     filename="MaterialHandling.spp", modelname=frame)

        # Step 4- Run the simulation

        my_plant.process_simulation()
        print(my_method.r_episode)

        assert isinstance(my_method.r_episode, np.ndarray)
        assert my_method.r_episode.all() == 0

    @patch("win32com.client.Dispatch")
    def test_example_2(self, dispatch):
        import simulai

        # Step 0 - Definition of model frame

        frame = "Modelo"

        # Step 1- Definition of input variables

        espera = simulai.DiscreteVariable(
            "Espera", 60, 300, 10, "Models."+frame+".espera")
        stock = simulai.DiscreteVariable(
            "Stock", 10, 40, 10, "Models."+frame+".stock")
        numviajes = simulai.DiscreteVariable(
            "Numero de viajes", 1, 4, 1, "Models."+frame+".numviajes")

        var_input = [espera, stock, numviajes]

        # Step 2- Definition of output variables

        transportes = simulai.OutcomeVariable(
            "Distancia Transportes", "Models."+frame+".transportes", 2, 9)
        buffers = simulai.OutcomeVariable(
            "Llenado buffers", "Models."+frame+".buffers", 3, 20)
        salidas = simulai.OutcomeVariable(
            "Espera en las Salidas", "Models."+frame+".salidas", 2, 20)

        var_output = [transportes, buffers, salidas]

        # Step 3- Choice of method, plant and simulation file.
        # Enter input variables and output variables.

        my_method = simulai.Sarsa(v_i=var_input, episodes_max=10, steps_max=10)
        my_plant = simulai.BasePlant(method=my_method, v_i=var_input, v_o=var_output,
                                     filename="MaterialHandling.spp", modelname=frame)

        # Step 4- Run the simulation

        my_plant.process_simulation()
        print(my_method.r_episode)

        assert isinstance(my_method.r_episode, np.ndarray)
        assert my_method.r_episode.all() == 0