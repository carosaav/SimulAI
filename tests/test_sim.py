import pytest
import numpy as np

from numpy.testing import assert_equal

from simulai import sim

from abc import ABCMeta, abstractmethod


@pytest.fixture
def var_input():
    frame = "Modelo"
    espera = sim.DiscreteVariable(
        "Espera", 60, 300, 10, "Models." + frame + ".espera")
    stock = sim.DiscreteVariable(
        "Stock", 10, 50, 10, "Models." + frame + ".stock")
    numviajes = sim.DiscreteVariable(
        "Numero de viajes", 1, 5, 1, "Models." + frame + ".numviajes"
    )
    vi = [espera, stock, numviajes]

    return vi


@pytest.fixture
def var_out():
    frame = "Modelo"
    transportes = sim.OutcomeVariable(
        "Distancia Transportes", "Models." + frame + ".transportes", 2, 9
    )
    buffers = sim.OutcomeVariable(
        "Llenado buffers", "Models." + frame + ".buffers", 3, 20
    )
    salidas = sim.OutcomeVariable(
        "Espera en las Salidas", "Models." + frame + ".salidas", 2, 20
    )
    vo = [transportes, buffers, salidas]

    return vo


@pytest.fixture
def base(var_input, var_out):
    plant = sim.BasePlant(
        method=sim.Qlearning(v_i=var_input, episodes_max=1, steps_max=10),
        v_i=var_input,
        v_o=var_out,
        filename="MaterialHandling.spp",
        modelname="Model",
    )

    return plant

@pytest.fixture
def my_method(var_input):
    method = sim.Qlearning(v_i=var_input, episodes_max=1, steps_max=10)

    return method

def test_DiscreteVariable():
    parm = sim.DiscreteVariable("Susan", 0, 10, 1, "path")

    assert isinstance(parm.name, str)
    assert isinstance(parm.lower_limit, int)
    assert isinstance(parm.upper_limit, int)
    assert isinstance(parm.step, int)
    assert isinstance(parm.path, str)


def test_OutcomeVariable():
    parm = sim.OutcomeVariable("Susan", "path", 5, 1)

    assert isinstance(parm.name, str)
    assert isinstance(parm.path, str)
    assert isinstance(parm.column, int)
    assert isinstance(parm.num_rows, int)


def test_arg_BasePlant(base):

    assert isinstance(base.v_i, list)
    assert isinstance(base.v_o, list)
    assert isinstance(base.filename, str)
    assert isinstance(base.modelname, str)


def test_get_file_name_plant(base):

    filename = base.get_file_name_plant()

    assert filename == "MaterialHandling.spp"


def test_update(base, var_input):
    assert isinstance(base.r_episode, int)


def test_process_simulation(base):
    connect = base.connection()

    assert isinstance(connect, bool)

def test_ini_saq(my_method):
    """Test that the output Q matrix has the necessary characteristics.

    Initially the dimensions are checked.
    Then it is verified that the matrix is composed by 0 (zeros).
    """
    Q, S, A = my_method.ini_saq()

    assert (isinstance(Q, np.ndarray))
    assert (isinstance(S, np.ndarray))
    assert (isinstance(A, np.ndarray))
    assert Q.shape == (625, 27)
    assert np.all((Q == 0)) == True
    assert S.shape == (625, 3)
    assert np.all((S == 0)) == False
    assert A.shape == (27, 3)
    assert np.all((A == 0)) == False


def test_default_Q(var_input, var_out):
    parm = sim.Qlearning(v_i=var_input)

    assert isinstance(parm.s, list)
    assert isinstance(parm.a, list)
    assert isinstance(parm.v_i, list)
    assert isinstance(parm.alfa, float)
    assert isinstance(parm.gamma, float)
    assert isinstance(parm.epsilon, float)
    assert isinstance(parm.episodes_max, int)
    assert isinstance(parm.steps_max, int)
    assert isinstance(parm.r_episode, np.ndarray)
    assert_equal(parm.s, [])
    assert_equal(parm.a, [])
    assert_equal(parm.alfa, 0.10)
    assert_equal(parm.gamma, 0.90)
    assert_equal(parm.epsilon, 0.10)
    assert_equal(parm.episodes_max, 100)
    assert_equal(parm.steps_max, 100)

def test_arrays(my_method):
    arr = my_method.arrays()

    assert (isinstance(arr.s, np.ndarray))

def test_choose_action(var_input):
    method = sim.Qlearning(
        v_i=var_input, episodes_max=1, steps_max=10, seed=24)
    QSA = method.ini_saq()
    i = method.choose_action(np.random.randint(624))

    assert_equal(i, 0)

def test_process(my_method):
    r = my_method.process()

    assert (isinstance(r, np.ndarray))





