# This file is part of the SimulAI Project
# (https://github.com/carosaav/SimulAI).
# Copyright (c) 2020, Perez Colo Ivo, Pirozzo Bernardo Manuel,
# Carolina Saavedra Sueldo
# License: MIT
# Full Text: https://github.com/carosaav/SimulAI/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================


import win32com.client as win32
import os


# ============================================================================
# COMMUNICATION
# ============================================================================

def check_connection(method):
    """Check the connection status, returning an error.

    menssaje if it fails
    """
    def wrapper(self, *args, **kwargs):
        if not self.is_connected:
            raise ConnectionError("Not connected", self.model_name)
        output = method(self, *args, **kwargs)
        return output
    return wrapper


class Com(object):
    """Definition of the function of communication.

    Parameters
    ----------
    model_name: str
              Is the model_name of the file
    Attributes
    ----------
    is_connected: bool
                Connection status
    """

    def __init__(self, model_name):
        self.model_name = model_name        
        self.is_connected = False
        self.plant_simulation = ""

    def get_path_file_model(self):
        """Return the complete file path.

        Input parameter: file name
        Return: file path
        """
        path = os.getcwd() + "\\" + self.model_name
        return path

    def connection(self):
        """Return the connection object.

        Input parameter: file name
        Return: connection object
        """
        path_file = self.get_path_file_model()
        try:
            self.plant_simulation = win32.Dispatch(
                "Tecnomatix.PlantSimulation.RemoteControl.15.0")
            self.plant_simulation.loadModel(path_file)
            print("The connection was successful")
            self.is_connected = True
            return True
        except ConnectionError:
            print("Connection error. path file: " + path_file)
            return False

    @check_connection
    def setVisible(self, value):
        """Execute the application Tecnomatix.

        Input parameter: model_name and value
        """
        self.plant_simulation.setVisible(value)

    @check_connection
    def setValue(self, ref, value):
        """Set the values in the simulator.

        Input parameter: model_name, ref and value
        """
        self.plant_simulation.setValue(ref, value)

    @check_connection
    def getValue(self, ref):
        """Get the values in the simulator.

        Input parameter: model_name, ref and value
        """
        return self.plant_simulation.getValue(ref)

    @check_connection
    def startSimulation(self, ref):
        """Make the simulation start.

        Input parameter: model_name and ref
        """
        self.plant_simulation.startSimulation(ref)

    @check_connection
    def resetSimulation(self, ref):
        """
        Make the simulation reset.

        Input parameter: model_name and ref
        """
        self.plant_simulation.resetSimulation(ref)

    @check_connection
    def stopSimulation(self, ref):
        """Make the simulation stop.

        Input parameter: model_name and ref
        """
        self.plant_simulation.stopSimulation(ref)

    @check_connection
    def closeModel(self):
        """Close the simulation model.

        Input parameter: model_name
        """
        self.plant_simulation.CloseModel()

    @check_connection
    def executeSimTalk(self, ref, value):
        """Execute the simulation call.

        Input parameter: name_model, ref and value
        """
        self.plant_simulation.ExecuteSimTalk(ref, value)

    @check_connection
    def isSimulationRunning(self):
        """Check if the simulation is running.

        Input parameter: model_name
        """
        return(self.plant_simulation.IsSimulationRunning())

    @check_connection
    def loadModel(self, ref, value):
        """Perform the load of the model.

        Input parameter: model_name, ref and value
        """
        self.plant_simulation.LoadModel(ref, value)

    @check_connection
    def newModel(self):
        """Create a new model.

        Input parameter: model_name
        """
        self.plant_simulation.NewModel()

    @check_connection
    def openConsoleLogFile(self, ref):
        """Open the simulation result in the console.

        Input parameter: model_name and ref
        """
        self.plant_simulation.OpenConsoleLogFile(ref)

    @check_connection
    def quit(self):
        """Clear all result.

        Input parameter: model_name
        """
        self.plant_simulation.Quit()

    @check_connection
    def quitAfterTime(self, value):
        """Clear all result after a time.

        Input parameter: model_name and value
        """
        self.plant_simulation.QuitAfterTime(value)

    @check_connection
    def saveModel(self, ref):
        """Save the model result.

        Input parameter: model_name and ref
        """
        self.plant_simulation.SaveModel(ref)

    @check_connection
    def setLicenseType(self, ref):
        """Set the type of license.

        Input parameter: model_name and ref
        """
        self.plant_simulation.SetLicenseType(ref)

    @check_connection
    def setNoMessageBox(self, value):
        """Delete the messages on the screen.

        Input parameter: model_name and value
        """
        self.plant_simulation.SetNoMessageBox(value)

    @check_connection
    def setPathContext(self, ref):
        """Set the context.

        Input parameter: model_name and ref
        """
        self.plant_simulation.SetPathContext(ref)

    @check_connection
    def setSuppressStartOf3D(self, value):
        """Eliminate the start of 3D model.

        Input parameter: model_name and value
        """
        self.plant_simulation.SetSuppressStartOf3D(value)

    @check_connection
    def setTrustModels(self, value):
        """Set the real model.

        Input parameter: model_name and value
        """
        self.plant_simulation.SetTrustModels(value)

    @check_connection
    def transferModel(self, value):
        """Transfer the model.

        Input parameter: model_name and value
        """
        self.plant_simulation.TransferModel(value)
