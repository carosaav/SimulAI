

import plant as pl
import autonomous_decision_system as ads


METHODS = {"Q_learning": ads.Q_learning()}


def plant_simulation_node(m="Q_learning", filename="MaterialHandling.spp"):

    method = METHODS[m]
    plant = pl.Plant_1(method, filename)
    plant.process_simulation()


if __name__ == '__main__':
    plant_simulation_node()
