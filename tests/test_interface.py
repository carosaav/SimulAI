
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
        return simulai.CommunicationInterface('ModelName.spp')

    # =================================================
    # Now the actual testing
    # =================================================

    @patch('win32com.client.Dispatch')
    def test_connection_valid_name(self, dispatch, com):
        # Test as if a valid Model Name was given
        # Dispatch will not raise any exceptions
        com.connection()
        dispatch.assert_called_once()

    @patch('win32com.client.Dispatch', side_effect=Exception('EXCEPTION'))
    def test_connection_invalid_name(self, dispatch, com):
        # Test as if an invalid Model Name was given
        # Dispatch will raise an Exception
        with pytest.raises(Exception) as err:
            com.connection()

        # The initial plant_simulation value was an empty string
        assert isinstance(com.plant_simulation, str)

    @patch('win32com.client.Dispatch')
    def test_setvisible(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.setvisible(True)
        com.plant_simulation.setVisible.assert_called_with(True)

    @patch('win32com.client.Dispatch')
    def test_setvalue(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.setvalue('foo', 42)
        com.plant_simulation.setValue.assert_called_with('foo', 42)

    @patch('win32com.client.Dispatch')
    def test_getvalue(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.getvalue('foo')
        com.plant_simulation.getValue.assert_called_with('foo')

    @patch('win32com.client.Dispatch')
    def test_startsimulation(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.startsimulation('foo')
        com.plant_simulation.startSimulation.assert_called_with('foo')

    @patch('win32com.client.Dispatch')
    def test_resetsimulation(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.resetsimulation('foo')
        com.plant_simulation.resetSimulation.assert_called_with('foo')

    @patch('win32com.client.Dispatch')
    def test_stopsimulation(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.stopsimulation('foo')
        com.plant_simulation.stopSimulation.assert_called_with('foo')

    @patch('win32com.client.Dispatch')
    def test_closemodel(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.closemodel()
        com.plant_simulation.CloseMmodel.assert_called_with()

    @patch('win32com.client.Dispatch')
    def test_execute_simtalk(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.execute_simtalk('foo', 24)
        com.plant_simulation.ExecuteSimTalk.assert_called_with('foo', 24)

    @patch('win32com.client.Dispatch')
    def test_is_simulation_running(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.is_simulation_running()
        com.plant_simulation.IsSimulationRunning.assert_called_with()

    @patch('win32com.client.Dispatch')
    def test_loadmodel(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.loadmodel('foo', 24)
        com.plant_simulation.LoadModel.assert_called_with('foo', 24)

    @patch('win32com.client.Dispatch')
    def test_newmodel(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.newmodel()
        com.plant_simulation.NewModel.assert_called_with()

    @patch('win32com.client.Dispatch')
    def test_openconsole_logfile(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.openconsole_logfile('foo')
        com.plant_simulation.OpenConsoleLogFile.assert_called_with('foo')

    @patch('win32com.client.Dispatch')
    def test_quit(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.quit()
        com.plant_simulation.Quit.assert_called_once_with()

    @patch('win32com.client.Dispatch')
    def test_quit_aftertime(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.quit_aftertime(24)
        com.plant_simulation.QuitAfterTime.assert_called_with(24)

    @patch('win32com.client.Dispatch')
    def test_savemodel(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.savemodel('foo')
        com.plant_simulation.SaveModel.assert_called_with('foo')

    @patch('win32com.client.Dispatch')
    def test_set_licensetype(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.set_licensetype('foo')
        com.plant_simulation.SetLicenseType.assert_called_with('foo')

    @patch('win32com.client.Dispatch')
    def test_set_no_messagebox(self, dispatch, com):
        com.connection()
        com.plant_simulation.assert_called_once()

        com.set_no_messagebox(24)
        com.plant_simulation.SetNoMessageBox.assert_called_with(24)

    @patch('win32com.client.Dispatch')
    def test_set_pathcontext(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.set_pathcontext('foo')
        com.plant_simulation.SetPathContext.assert_called_with('foo')

    @patch('win32com.client.Dispatch')
    def test_set_suppress_start_of_3d(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.set_suppress_start_of_3d(24)
        com.plant_simulation.SetSuppressStartOf3D.assert_called_with(24)

    @patch('win32com.client.Dispatch')
    def test_set_trustmodels(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.set_trustmodels(24)
        com.plant_simulation.SetTrustModels.assert_called_with(24)

    @patch('win32com.client.Dispatch')
    def test_transfermodel(self, dispatch, com):
        com.connection()
        dispatch.assert_called_once()

        com.transfermodel(24)
        com.plant_simulation.TransferModel.assert_called_with(24)
