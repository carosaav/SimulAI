import pytest
import numpy as np

from simulai import sim

@pytest.fixture
def var_input():
	frame = "Modelo"
	espera = sim.DiscreteVariable(
		"Espera", 60, 300, 10, "Models."+frame+".espera")
	stock = sim.DiscreteVariable(
		 	"Stock", 10, 50, 10, "Models."+frame+".stock")
	numviajes = sim.DiscreteVariable(
		"Numero de viajes", 1, 5, 1, "Models."+frame+".numviajes")
	vi = [espera, stock, numviajes]
	
	return vi

#def test_arrays(var_input):


def test_ini_saq(var_input):
	"""Test that the output Q matrix has the necessary characteristics. 

	Initially the dimensions are checked.
	Then it is verified that the matrix is composed by 0 (zeros).
	"""
	my_method = sim.Qlearning(v_i=var_input, episodes_max=1, steps_max=10)
	Q, S, A = my_method.ini_saq()

	assert Q.shape == (625, 27) 

	assert np.all((Q == 0)) == True

	assert S.shape == (625, 3)

	assert np.all((S == 0)) == False

	assert A.shape == (27, 3)

	assert np.all((A == 0)) == False

def test_choose_action(var_input):
	my_method = sim.Qlearning(v_i=var_input, episodes_max=1, steps_max=10, seed=24)
	_ = my_method.ini_saq()
	
	assert my_method.choose_action(np.random.randint(624)) == 0

def test_process(var_input):
	my_method = sim.Qlearning(v_i=var_input, episodes_max=1, steps_max=10)
	Q, S, A = my_method.ini_saq()
	r = my_method.process()

	assert my_method.process == 0