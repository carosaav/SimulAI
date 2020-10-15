

import plant as pl
import autonomous_decision_system as ads


def plant_simulation_node():

    method = ads.Q_learning()
    plant = pl.Plant_1(method)
    plant.process_simulation()


if __name__ == '__main__':
    plant_simulation_node()
