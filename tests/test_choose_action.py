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

def test_ini_saq(var_input):
	my_method = sim.Qlearning(v_i=var_input, episodes_max=1, steps_max=10)
	S, A, Q = my_method.ini_saq()

	assert S.shape == (625, 3)

def test_choose_action(var_input):
	my_method = sim.Qlearning(v_i=var_input, episodes_max=1, steps_max=10)
	_ = my_method.ini_saq()

	assert my_method.choose_action(1) == 0
