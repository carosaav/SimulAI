
# This file is part of the
# SimulAI Project (https://github.com/carosaav/SimulAI).
# Copyright (c) 2020, Perez Colo Ivo, Pirozzo Manuel Bernardo,
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
    """
    This function checks the connection status, returning an error
    menssaje if it fails
    """
    def wrapper(self, *args, **kwargs):
        if self.is_connected is True:
            self(*args, **kwargs)
        else:
            raise ConnectionError("Not connected:", self.name)
        output = method(self, *args, **kwargs)
        return output
    return wrapper

class Com(object):
    """
    Definition of the function of communication
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.is_connected = False


    def get_path_file_model(self):
        """
        Function that returns the complete file path
        Input parameter: file name
        Return: file path
        """
        path = os.getcwd() + "\\" + self.model_name
        return path

    
    def connection(self):
        """
        Function that returns the connection object
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
        except:
            print(("Connection error. path file: " + path_file))
            return False

    @check_connection
    def setVisible(self, value):
        self.plant_simulation.setVisible(value)

    @check_connection
    def setValue(self, ref, value):
        self.plant_simulation.setValue(ref, value)

    @check_connection
    def getValue(self, ref):
        return self.plant_simulation.getValue(ref)

    @check_connection
    def startSimulation(self, ref):
        self.plant_simulation.startSimulation(ref)
        
    @check_connection
    def resetSimulation(self, ref):
        self.plant_simulation.resetSimulation(ref)

    @check_connection
    def stopSimulation(self, ref):
        self.plant_simulation.stopSimulation(ref)
    
    @check_connection
    def closeModel(self):
        self.plant_simulation.CloseModel()
        
    @check_connection
    def executeSimTalk(self, ref, value):
        self.plant_simulation.ExecuteSimTalk(ref, value)
    
    @check_connection
    def isSimulationRunning(self):
        return(self.plant_simulation.IsSimulationRunning())        

    @check_connection    
    def loadModel(self, ref, value):
        self.plant_simulation.LoadModel(ref, value)

    @check_connection    
    def newModel(self):
        self.plant_simulation.NewModel()
    
    @check_connection    
    def openConsoleLogFile(self, ref):
        self.plant_simulation.OpenConsoleLogFile(ref)

    @check_connection   
    def quit(self):
        self.plant_simulation.Quit()

    @check_connection    
    def quitAfterTime(self, value):
        self.plant_simulation.QuitAfterTime(value)

    @check_connection    
    def saveModel(self, ref):
        self.plant_simulation.SaveModel(ref)

    @check_connection    
    def setLicenseType(self, ref):
        self.plant_simulation.SetLicenseType(ref)

    @check_connection    
    def setNoMessageBox(self, value):
        self.plant_simulation.SetNoMessageBox(value)

    @check_connection    
    def setPathContext(self, ref):
            self.plant_simulation.SetPathContext(ref)

    @check_connection        
    def setSuppressStartOf3D(self, value):
        self.plant_simulation.SetSuppressStartOf3D(value)

    @check_connection    
    def setTrustModels(self, value):
        self.plant_simulation.SetTrustModels(value)

    @check_connection    
    def transferModel(self, value):
        self.plant_simulation.TransferModel(value)
