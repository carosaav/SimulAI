import pytest
import numpy as np

from numpy.testing import assert_equal, assert_


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

@pytest.fixture
def var_out():
	frame = "Modelo"
	transportes = sim.OutcomeVariable(
		"Distancia Transportes", "Models."+frame+".transportes", 2, 9)
	buffers = sim.OutcomeVariable(
		"Llenado buffers", "Models."+frame+".buffers", 3, 20)
	salidas = sim.OutcomeVariable(
		"Espera en las Salidas", "Models."+frame+".salidas", 2, 20)
	vo = [transportes, buffers, salidas]

	return vo

def test_DiscreteVariable():
	parm = sim.DiscreteVariable("Susan", 0, 10, 1, "path")

	assert_(isinstance(parm.name, str))
	assert_(isinstance(parm.lower_limit, int))
	assert_(isinstance(parm.upper_limit, int))
	assert_(isinstance(parm.step, int))
	assert_(isinstance(parm.path, str))

def test_OutcomeVariable():
	parm = sim.OutcomeVariable("Susan", "path", 5, 1)

	assert_(isinstance(parm.name, str))
	assert_(isinstance(parm.path, str))
	assert_(isinstance(parm.column, int))
	assert_(isinstance(parm.num_rows, int))

#class test_BasePlant():
funct = sim.BasePlant(method=sim.Qlearning(
		v_i=var_input, episodes_max=1, steps_max=10), v_i=var_input,
		v_o=var_out, filename="MaterialHandling.spp", modelname="Model")
	
def test_arg_BasePlant():
	
	#assert(isinstance(parm.v_i, list))
	#assert(isinstance(parm.v_o, list))
	assert_(isinstance(funct.filename, str))
	assert_(isinstance(funct.modelname, str))

def test_get_file_name_plant():
	
	filename = funct.get_file_name_plant()

	assert filename == "MaterialHandling.spp"

def test_update(): 
	updt = funct.update(data=np.array(5)) 

	assert_(isinstance(updt.r_episode, int))




def test_default_Q(var_input, var_out):
	"""Test that the default values are correct"""
	parm = sim.Qlearning(v_i=var_input)

	assert_equal(parm.alfa, 0.10)
	assert_equal(parm.gamma, 0.90)
	assert_equal(parm.epsilon, 0.10)
	assert_equal(parm.episodes_max, 100)
	assert_equal(parm.steps_max, 100)


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








