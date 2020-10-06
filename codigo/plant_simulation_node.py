

from . import Material_Handling as M_A
from . import RL_Method_1 as RL


def plant_simulation_node():

    method = RL.RL_Method_1()
    plant = M_A.Material_Handling(method)
    plant.process_simulation()


if __name__ == '__main__':
    plant_simulation_node()
