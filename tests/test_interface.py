
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
import simulai


# ============================================================================
# TESTS
# ============================================================================

@pytest.fixture
def base():
    com = simulai.CommunicationInterface("MaterialHandling.spp")

    return com


@pytest.fixture
def base2():
    com = simulai.CommunicationInterface("aaa")

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
        import simulai
        return simulai.CommunicationInterface('MaterialHandling.spp')

    # =================================================
    # Now the actual testing
    # =================================================

    @patch('win32com.client.Dispatch')
    def test_connection_valid_name(self, dispatch, com):
        # Test as if a valid Model Name was given
        # Dispatch will not raise any exceptions
        com.connection()
        dispatch.assert_called_once()

    # @patch('win32com.client.Dispatch')
    # def test_valid_loadModel(self, dispatch, com):
    #     # Test as if a valid Model Name was given
    #     # Dispatch will not raise any exceptions
    #     com.connection()
    #     dispatch.loadModel.assert_called_once_with('path')

    # @patch('win32com.client.Dispatch')
    # def test_setVisible(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.setVisible(True)
    #     dispatch.setVisible.assert_called_once_with(True)

    # @patch('win32com.client.Dispatch')
    # def test_setValue(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.setValue('foo', 24)
    #     dispatch.setValue.assert_called_once_with('foo', 24)

    # @patch('win32com.client.Dispatch')
    # def test_getValue(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.getValue('foo')
    #     dispatch.getValue.assert_called_once_with('foo')

    # @patch('win32com.client.Dispatch')
    # def test_startSimulation(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.startSimulation('foo')
    #     dispatch.startSimulation.assert_called_once_with('foo')

    # @patch('win32com.client.Dispatch')
    # def test_resetSimulation(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.resetSimulation('foo')
    #     dispatch.resetSimulation.assert_called_once_with('foo')

    # @patch('win32com.client.Dispatch')
    # def test_stopSimulation(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.stopSimulation('foo')
    #     dispatch.stopSimulation.assert_called_once_with('foo')

    # @patch('win32com.client.Dispatch')
    # def test_closeModel(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.closeModel()
    #     dispatch.CloseModel.assert_called_once_with()

    # @patch('win32com.client.Dispatch')
    # def test_executeSimTalk(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.executeSimTalk('foo', 24)
    #     dispatch.ExecuteSimTalk.assert_called_once_with('foo', 24)

    # @patch('win32com.client.Dispatch')
    # def test_isSimulationRunning(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.isSimulationRunning()
    #     dispatch.IsSimulationRunning.assert_called_once_with()

    # @patch('win32com.client.Dispatch')
    # def test_loadModel(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.loadModel('foo', 24)
    #     dispatch.LoadModel.assert_called_once_with('foo', 24)

    # @patch('win32com.client.Dispatch')
    # def test_newModel(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.newModel()
    #     dispatch.NewModel.assert_called_once_with()

    # @patch('win32com.client.Dispatch')
    # def test_openConsoleLogFile(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.openConsoleLogFile('foo')
    #     dispatch.OpenConsoleLogFile.assert_called_once_with('foo')

    # @patch('win32com.client.Dispatch')
    # def test_quit(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.quit()
    #     dispatch.Quit.assert_called_once_with()

    # @patch('win32com.client.Dispatch')
    # def test_quitAfterTime(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.quitAfterTime(24)
    #     dispatch.QuitAfterTime.assert_called_once_with(24)

    # @patch('win32com.client.Dispatch')
    # def test_saveModel(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.saveModel('foo')
    #     dispatch.SaveModel.assert_called_once_with('foo')

    # @patch('win32com.client.Dispatch')
    # def test_setLicenseType(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.setLicenseType('foo')
    #     dispatch.SetLicenseType.assert_called_once_with('foo')

    # @patch('win32com.client.Dispatch')
    # def test_setNoMessageBox(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.setNoMessageBox(24)
    #     dispatch.SetNoMessageBox.assert_called_once_with(24)

    # @patch('win32com.client.Dispatch')
    # def test_setPathContext(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.setPathContext('foo')
    #     dispatch.SetPathContext.assert_called_once_with('foo')

    # @patch('win32com.client.Dispatch')
    # def test_setSuppressStartOf3D(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.setSuppressStartOf3D(24)
    #     dispatch.SetSuppressStartOf3D.assert_called_once_with(24)

    # @patch('win32com.client.Dispatch')
    # def test_setTrustModels(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.setTrustModels(24)
    #     dispatch.SetTrustModels.assert_called_once_with(24)

    # @patch('win32com.client.Dispatch')
    # def test_transferModel(self, dispatch, com):
    #     com.connection()
    #     dispatch.loadModel.assert_called_once()

    #     com.transferModel(24)
    #     dispatch.TransferModel.assert_called_once_with(24)
