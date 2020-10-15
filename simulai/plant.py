

from abc import ABCMeta, abstractmethod
from communication_interface import Communication_Interface as C_I
import numpy as np


class Plant(metaclass=ABCMeta):
    def __init__(self, method):
        self.method = method
        method.register(self)

    def connection(self):
        file_name = self.get_file_name_plant()
        self.connect = C_I(file_name)
        return self.connect.connection()

    @abstractmethod
    def get_file_name_plant(self):
        pass

    @abstractmethod
    def process_simulation(self):
        pass

    @abstractmethod
    def update(self, data):
        pass

class Plant_1(Plant):

    def __init__(self, method):
        Plant.__init__(self, method)

    def get_file_name_plant(self):
        return "MaterialHandling.spp"

    def update(self, data):
        self.connect.setValue(".Models.Modelo.espera", data)
        self.connect.startSimulation(".Models.Modelo")

        a = np.zeros(20)
        b = np.zeros(20)
        for h in range(1, 21):
            a[h-1] = self.connect.getValue(
                ".Models.Modelo.buffers[3,%s]" % (h))
            b[h-1] = self.connect.getValue(
                ".Models.Modelo.salidas[2,%s]" % (h))
        c = np.sum(a)
        d = np.sum(b)
        r = c*0.5+d*0.5

        self.connect.resetSimulation(".Models.Modelo")
        return r

    def process_simulation(self):
        if (self.connection()):
            self.connect.setVisible(True)
            self.method.process()
            self.method.plot()
