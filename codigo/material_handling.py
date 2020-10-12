

from plant import Plant
import numpy as np


class Material_Handling(Plant):

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
