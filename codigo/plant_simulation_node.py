
from Autonomous_Decision_System import Autonomous_Decision_System
from material_handling import Material_Handling as M_A
from rl_method_1 import RL_Method_1 as RL

def valores():
	alfa = 0.0
	gamma = 0.0
	epsilon = 0.0
	ep = 0
	t = 0.0
	input('ingrese el valor de alfa ', alfa)
	input('ingrese el valor de gamma ', gamma)
	input('ingrese el valor de epsilon ', epsilon)
	input('ingrese el valor de episodios ', ep)
	input('ingrese el valor de tiempo máximo ', t)

def plant_simulation_node(alfa, gamma, epsilon, ep, t):

    method = RL(Autonomous_Decision_System, alfa, gamma, epsilon, ep, t)
    plant = M_A(method)
    plant.process_simulation()


if __name__ == '__main__':
    plant_simulation_node()
