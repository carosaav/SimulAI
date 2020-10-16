

import plant as pl
import autonomous_decision_system as ads


PLANTS = {"Plant_1": pl.Plant_1()}
METHODS = {"Q_learning": ads.Q_learning()}


class DiscreteVariable:
	"""This class allows you to enter the chosen Tecnomatix Plant Simulation
	variables as input to the artificial intelligence method. 
	Up to 4 discrete variables are allowed that must conform up to 625
	possible states. For example, if 4 variables are chosen, each of them
	can take 5 possible values.
	"""

	def __init__(self, name, lower_limit, upper_limit, step, path):
		self.name = name
		self.lower_limit = lower_limit
		self.upper_limit = upper_limit
		self.step = step
		self.path = path


class OutcomeVariable:
	"""This class allows you to enter the chosen Tecnomatix Plant Simulation
	variables as output to the artificial intelligence method.
	The chosen column from which to extract the results and the number of
	rows it has must be indicated
	"""
	def __init__(self, name, path, column, num_rows):
		self.name = name
		self.path = path
		self.column = column
		self.num_rows = num_rows


def plant_simulation_node(m, p, filename):

    method = METHODS[m]
    plant = PLANTS[p](method, filename)
    plant.process_simulation()


if __name__ == '__main__':
    plant_simulation_node()
