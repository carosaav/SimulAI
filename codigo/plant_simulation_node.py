

from material_handling import Material_Handling as M_A
from rl_method_1 import RL_Method_2 as RL


def plant_simulation_node():

    method = RL()
    plant = M_A(method)
    plant.process_simulation()


if __name__ == '__main__':
    plant_simulation_node()
