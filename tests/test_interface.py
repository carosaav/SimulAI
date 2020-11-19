
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
from unittest.mock import patch, MagicMock
from numpy.testing import assert_equal
from simulai import interface
import sys


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
            "win32com.client.Dispatch": self.win32com.client.Dispatch
            }

        self.module_patcher = patch.dict("sys.modules", mock_modules)
        self.module_patcher.start()

    def teardown_method(self):
        self.module_patcher.stop()

    @pytest.fixture
    def com(self):
        # Here we import interface with win32 patched
        from simulai import interface
        return interface.Com('ModelName.spp')

    # =================================================
    # Now the actual testing
    # =================================================

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_valid_loadModel(self, dispatch, com):
        # Test as if a valid Model Name was given
        # Dispatch will not raise any exceptions
        com.connection()
        com.plant_simulation.loadModel.assert_called_once_with('path')

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_setVisible(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.setVisible(True)
        com.plant_simulation.setVisible.assert_called_once_with(True)

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_setValue(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.setValue('foo', 24)
        com.plant_simulation.setValue.assert_called_once_with('foo', 24)

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_getValue(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.getValue('foo')
        com.plant_simulation.getValue.assert_called_once_with('foo')

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_startSimulation(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.startSimulation('foo')
        com.plant_simulation.startSimulation.assert_called_once_with('foo')

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_resetSimulation(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.resetSimulation('foo')
        com.plant_simulation.resetSimulation.assert_called_once_with('foo')

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_stopSimulation(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.stopSimulation('foo')
        com.plant_simulation.stopSimulation.assert_called_once_with('foo')

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_closeModel(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.closeModel()
        com.plant_simulation.CloseModel.assert_called_once_with()

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_executeSimTalk(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.executeSimTalk('foo', 24)
        com.plant_simulation.ExecuteSimTalk.assert_called_once_with('foo', 24)

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_isSimulationRunning(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        return_value = com.isSimulationRunning()
        com.plant_simulation.IsSimulationRunning.assert_called_once_with()

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_loadModel(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.loadModel('foo', 24)
        com.plant_simulation.LoadModel.assert_called_once_with('foo', 24)

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_newModel(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.newModel()
        com.plant_simulation.NewModel.assert_called_once_with()

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_openConsoleLogFile(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.openConsoleLogFile('foo')
        com.plant_simulation.OpenConsoleLogFile.assert_called_once_with('foo')

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_quit(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.quit()
        com.plant_simulation.Quit.assert_called_once_with()

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_quitAfterTime(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.quitAfterTime('foo')
        com.plant_simulation.QuitAfterTime.assert_called_once_with('foo')

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_quitAfterTime(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.quitAfterTime(24)
        com.plant_simulation.QuitAfterTime.assert_called_once_with(24)

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_saveModel(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.saveModel('foo')
        com.plant_simulation.SaveModel.assert_called_once_with('foo')

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_setLicenseType(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.setLicenseType('foo')
        com.plant_simulation.SetLicenseType.assert_called_once_with('foo')

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_setNoMessageBox(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.setNoMessageBox(24)
        com.plant_simulation.SetNoMessageBox.assert_called_once_with(24)

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_setPathContext(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.setPathContext('foo')
        com.plant_simulation.SetPathContext.assert_called_once_with('foo')

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_setSuppressStartOf3D(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.setSuppressStartOf3D(24)
        com.plant_simulation.SetSuppressStartOf3D.assert_called_once_with(24)

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_setTrustModels(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.setTrustModels(24)
        com.plant_simulation.SetTrustModels.assert_called_once_with(24)

    @pytest.mark.xfail
    @patch('win32com.client.Dispatch')
    def test_transferModel(self, dispatch, com):

        com.connection()
        com.plant_simulation.loadModel.assert_called_once()

        com.transferModel(24)
        com.plant_simulation.TransferModel.assert_called_once_with(24)