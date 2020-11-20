
# This file is part of the
#   SimulAI Project (https://github.com/carosaav/SimulAI).
# Copyright (c) 2020, Perez Colo Ivo, Pirozzo Bernardo Manuel,
# Carolina Saavedra Sueldo
# License: MIT
#   Full Text: https://github.com/carosaav/SimulAI/blob/master/LICENSE


# ============================================================================
# DOCS
# ============================================================================

"""Class makes the connection with Tecnomatix Plant Simulation."""

# ============================================================================
# IMPORTS
# ============================================================================

try:
    import win32com.client as win32
except ModuleNotFoundError:
    print("Install pywin32")
import os
from functools import wraps
import pathlib
import attr


# ============================================================================
# COMMUNICATION
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))


class ConnectionError(Exception):
    """Connection failed exception."""

    pass


class ModelNotFoundError(FileNotFoundError):
    """Custom error for Not found Model."""

    pass


def check_connection(method):
    """Check the connection status, returning an error.

    Parameters
    ----------
    method: str
        Name of the method.
    Return
    -------
    A message indicating failure.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.is_connected:
            raise ConnectionError("Not connected", self.model_name)
        output = method(self, *args, **kwargs)
        return output
    return wrapper


@attr.s
class CommunicationInterface(object):
    """Definition of the function of communication.

    Parameters
    ----------
    model_name: str
        Name of the Tecnomatix Plant Simulation file.

    Attributes
    ----------
    is_connected: bool
        Connection status
    plant_simulation: str
        Attribute for the return of the connection object.
    """

    model_name = attr.ib()
    is_connected = attr.ib(default=False)
    plant_simulation = attr.ib(default="")

    def get_path_file_model(self):
        """Return the complete file path.

        Return
        ------
        file path:str
            Path of Tecnomatix Plant Simulation file.
        """
        path = str(PATH / self.model_name)
        if not os.path.exists(path):
            raise ModelNotFoundError(
                f"Model {self.model_name} does not exists")
        return path

    def connection(self):
        """Return the connection object.

        Return
        ------
        connection status: bool
            Connection indicator.
        """
        path_file = self.get_path_file_model()
        self.plant_simulation = win32.Dispatch(
            "Tecnomatix.PlantSimulation.RemoteControl.15.0")
        self.plant_simulation.loadModel(path_file)
        print("The connection was successful")
        self.is_connected = True
        return True

    @check_connection
    def setvisible(self, value):
        """Execute the application Tecnomatix.

        Parameters
        ----------
        value: bool
            User-selected value.
        """
        self.plant_simulation.setVisible(value)

    @check_connection
    def setvalue(self, ref, value):
        """Set the values in the simulator.

        Parameters
        ----------
        ref:str
            Path of the variable.
        value:int
            User-selected value.
        """
        self.plant_simulation.setValue(ref, value)

    @check_connection
    def getvalue(self, ref):
        """Get the values of the simulator.

        Parameters
        ----------
        ref:str
            Path of the variable.
        """
        return self.plant_simulation.getValue(ref)

    @check_connection
    def startsimulation(self, ref):
        """Make the simulation start.

        Parameters
        ----------
        ref:str
            Path of the model.
        """
        self.plant_simulation.startSimulation(ref)

    @check_connection
    def resetsimulation(self, ref):
        """Make the simulation reset.

        Parameters
        ----------
        ref:str
            Path of the model.
        """
        self.plant_simulation.resetSimulation(ref)

    @check_connection
    def stopsimulation(self, ref):
        """Make the simulation stop.

        Parameters
        ----------
        ref:str
            Path of the model.
        """
        self.plant_simulation.stopSimulation(ref)

    @check_connection
    def closemodel(self):
        """Close the simulation model."""
        self.plant_simulation.CloseModel()

    @check_connection
    def execute_simtalk(self, ref, value):
        """Execute the simulation programming language.

        Parameters
        ----------
        ref:str
            Path of the variable.
        value:int
            User-selected value.
        """
        self.plant_simulation.ExecuteSimTalk(ref, value)

    @check_connection
    def is_simulation_running(self):
        """Check if the simulation is running."""
        return(self.plant_simulation.IsSimulationRunning())

    @check_connection
    def loadmodel(self, ref, value):
        """Perform the load of the model.

        Parameters
        ----------
        ref:str
            Path of the file.
        value:int
            User-selected value.
        """
        self.plant_simulation.LoadModel(ref, value)

    @check_connection
    def newmodel(self):
        """Create a new model."""
        self.plant_simulation.NewModel()

    @check_connection
    def openconsole_logfile(self, ref):
        """Open the simulation result in the console.

        Parameters
        ----------
        ref:str
            Path of the file.
        """
        self.plant_simulation.OpenConsoleLogFile(ref)

    @check_connection
    def quit(self):
        """Clear all result."""
        self.plant_simulation.Quit()

    @check_connection
    def quit_aftertime(self, value):
        """Clear all result after a time.

        Parameters
        ----------
        value:int
            User-selected value.
        """
        self.plant_simulation.QuitAfterTime(value)

    @check_connection
    def savemodel(self, ref):
        """Save the model result.

        Parameters
        ----------
        ref:str
            Path of the file.
        """
        self.plant_simulation.SaveModel(ref)

    @check_connection
    def set_licensetype(self, ref):
        """Set the type of the license.

        Parameters
        ----------
        ref:str
            Path of the file.
        """
        self.plant_simulation.SetLicenseType(ref)

    @check_connection
    def set_no_messagebox(self, value):
        """Delete the messages on the screen.

        Parameters
        ----------
        value:int
            User-selected value.
        """
        self.plant_simulation.SetNoMessageBox(value)

    @check_connection
    def set_pathcontext(self, ref):
        """Set the context.

        Parameters
        ----------
        ref:str
            Path of the file.
        """
        self.plant_simulation.SetPathContext(ref)

    @check_connection
    def set_suppress_start_of_3d(self, value):
        """Eliminate the start of 3D model.

        Parameters
        ----------
        value:int
            User-selected value.
        """
        self.plant_simulation.SetSuppressStartOf3D(value)

    @check_connection
    def set_trustmodels(self, value):
        """Set the real model.

        Parameters
        ----------
        value:int
            User-selected value.
        """
        self.plant_simulation.SetTrustModels(value)

    @check_connection
    def transfermodel(self, value):
        """Transfer the model.

        Parameters
        ----------
        value:int
            User-selected value.
        """
        self.plant_simulation.TransferModel(value)
