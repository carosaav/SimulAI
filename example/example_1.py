

from simulai import sim


# Step 1- Definition of input variables

espera = sim.DiscreteVariable(
    "Espera", 60, 300, 10, ".Models.Modelo.espera")
stock = sim.DiscreteVariable(
    "Stock", 10, 50, 10, ".Models.Modelo.stock")
numviajes = sim.DiscreteVariable(
    "Numero de viajes", 1, 5, 1, ".Models.Modelo.numviajes")

var_input = [espera, stock, numviajes]

# Step 2- Definition of output variables

transportes = sim.OutcomeVariable(
    "Distancia Transportes", ".Models.Modelo.transportes", 2, 9)
buffers = sim.OutcomeVariable(
    "Llenado buffers", ".Models.Modelo.buffers", 3, 20)
salidas = sim.OutcomeVariable(
    "Espera en las Salidas", ".Models.Modelo.salidas", 2, 20)

var_output = [transportes, buffers, salidas]

# Step 3- Choice of method, plant and simulation file.
# Enter input variables and output variables.

my_method = sim.Qlearning(v_i=var_input, episodes_max=5, steps_max=10)
my_plant = sim.Plant1(method=my_method, filename="MaterialHandling.spp",
                      modelname="Modelo", v_i=var_input, v_o=var_output)

# Step 4- Run the simulation

example1 = sim.simulation_node(my_plant)
