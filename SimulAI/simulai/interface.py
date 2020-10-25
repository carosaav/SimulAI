

# ============================================================================
# IMPORTS
# ============================================================================


import win32com.client as win32
import os


# ============================================================================
# COMMUNICATION
# ============================================================================


class Com(object):

    def __init__(self, model_name):
        self.model_name = model_name
        self.is_connected = False
        self.plant_simulation = ""

    # Function that returns the complete file path
    # Input parameter: file name
    # Return: file path
    def get_path_file_model(self):
        path = os.getcwd() + "\\" + self.model_name
        return path

    # Function that returns the connection object
    # Input parameter: file name
    # Return: connection object
    def connection(self):
        path_file = self.get_path_file_model()
        try:
            self.plant_simulation = win32.Dispatch(
                "Tecnomatix.PlantSimulation.RemoteControl.15.0")
            self.plant_simulation.loadModel(path_file)
            print("The connection was successful")
            self.is_connected = True
            return True
        except:
            print(("Connection error. path file: " + path_file))
            return False

    #Decorator to verify connection
    def check_connetion(setVisible):
        def check(self,value):
            if self.is_connected is True:
                self.plant_simulation.setVisible(value)
            else:
                raise ConnectionError("Not connected")
        return(check)

    @check_connetion
    def setVisible(self, value):
        self.plant_simulation.setVisible(value)

    def setValue(self, ref, value):
        if self.is_connected is True:
            self.plant_simulation.setValue(ref, value)
        else:
            raise ConnectionError("Not connected")

    def getValue(self, ref):
        if self.is_connected is True:
            return self.plant_simulation.getValue(ref)
        else:
            raise ConnectionError("Not connected")

    def startSimulation(self, ref):
        if self.is_connected is True:
            self.plant_simulation.startSimulation(ref)
        else:
            raise ConnectionError("Not connected")

    def resetSimulation(self, ref):
        if self.is_connected is True:
            self.plant_simulation.resetSimulation(ref)
        else:
            raise ConnectionError("Not connected")

    def stopSimulation(self, ref):
        if self.is_connected is True:
            self.plant_simulation.stopSimulation(ref)
        else:
            raise ConnectionError("Not connected")

    def closeModel(self):
        if self.is_connected is True:
            self.plant_simulation.CloseModel()
        else:
            raise ConnectionError("Not connected")

    def executeSimTalk(self, ref, value):
        if self.is_connected is True:
            self.plant_simulation.ExecuteSimTalk(ref, value)
        else:
            raise ConnectionError("Not connected")

    def isSimulationRunning(self):
        if self.is_connected is True:
            return self.plant_simulation.IsSimulationRunning()
        else:
            raise ConnectionError("Not connected")

    def loadModel(self, ref, value):
        if self.is_connected is True:
            self.plant_simulation.LoadModel(ref, value)
        else:
            raise ConnectionError("Not connected")

    def newModel(self):
        if self.is_connected is True:
            self.plant_simulation.NewModel()
        else:
            raise ConnectionError("Not connected")

    def openConsoleLogFile(self, ref):
        if self.is_connected is True:
            self.plant_simulation.OpenConsoleLogFile(ref)
        else:
            raise ConnectionError("Not connected")

    def quit(self):
        if self.is_connected is True:
            self.plant_simulation.Quit()
        else:
            raise ConnectionError("Not connected")

    def quitAfterTime(self, value):
        if self.is_connected is True:
            self.plant_simulation.QuitAfterTime(value)
        else:
            raise ConnectionError("Not connected")

    def saveModel(self, ref):
        if self.is_connected is True:
            self.plant_simulation.SaveModel(ref)
        else:
            raise ConnectionError("Not connected")

    def setLicenseType(self, ref):
        if self.is_connected is True:
            self.plant_simulation.SetLicenseType(ref)
        else:
            raise ConnectionError("Not connected")

    def setNoMessageBox(self, value):
        if self.is_connected is True:
            self.plant_simulation.SetNoMessageBox(value)
        else:
            raise ConnectionError("Not connected")

    def setPathContext(self, ref):
        if self.is_connected is True:
            self.plant_simulation.SetPathContext(ref)
        else:
            raise ConnectionError("Not connected")

    def setSuppressStartOf3D(self, value):
        if self.is_connected is True:
            self.plant_simulation.SetSuppressStartOf3D(value)
        else:
            raise ConnectionError("Not connected")

    def setTrustModels(self, value):
        if self.is_connected is True:
            self.plant_simulation.SetTrustModels(value)
        else:
            raise ConnectionError("Not connected")

    def transferModel(self, value):
        if self.is_connected is True:
            self.plant_simulation.TransferModel(value)
        else:
            raise ConnectionError("Not connected")
