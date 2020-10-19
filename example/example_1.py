

import simulai


# Step 1- Definition of input variables

espera = simulai.DiscreteVariable(
    "Espera", 60, 300, 10, "Models.Modelo.espera")
stock = simulai.DiscreteVariable(
    "Stock", 10, 50, 10, ".Models.Modelo.stock")
numviajes = simulai.DiscreteVariable(
    "Numero de viajes", 1, 5, 1, ".Models.Modelo.numviajes")

var_input = [espera, stock, numviajes]

# Step 2- Definition of output variables

transportes = simulai.OutcomeVariable(
    "Distancia Transportes", ".Models.Modelo.transportes", 2, 9)
buffers = simulai.OutcomeVariable(
    "Llenado buffers", ".Models.Modelo.buffers", 3, 20)
salidas = simulai.OutcomeVariable(
    "Espera en las Salidas", ".Models.Modelo.salidas", 2, 20)

var_output = [transportes, buffers, salidas]

# Step 3- Choice of method, plant and simulation file.
# Enter input variables and output variables.

my_method = simulai.Q_learning(v_i=var_input, episodes_max=5, steps_max=10)
my_plant = simulai.Plant_1(method=my_method, filename="MaterialHandling.spp",
    v_i=var_input, v_o=var_output)

# Step 4- Run the simulation

example1 = simulai.plant_simulation_node(my_plant)
