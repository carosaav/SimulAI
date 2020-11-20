

import simulai

# Step 0 - Definition of model frame

frame = "Modelo"

# Step 1- Definition of input variables

espera = simulai.DiscreteVariable(
    "Espera", 60, 300, 10, "Models."+frame+".espera")
stock = simulai.DiscreteVariable(
    "Stock", 10, 40, 10, "Models."+frame+".stock")
numviajes = simulai.DiscreteVariable(
    "Numero de viajes", 1, 4, 1, "Models."+frame+".numviajes")

var_input = [espera, stock, numviajes]

# Step 2- Definition of output variables

transportes = simulai.OutcomeVariable(
    "Distancia Transportes", "Models."+frame+".transportes", 2, 9)
buffers = simulai.OutcomeVariable(
    "Llenado buffers", "Models."+frame+".buffers", 3, 20)
salidas = simulai.OutcomeVariable(
    "Espera en las Salidas", "Models."+frame+".salidas", 2, 20)

var_output = [transportes, buffers, salidas]

# Step 3- Choice of method, plant and simulation file.
# Enter input variables and output variables.

my_method = simulai.Sarsa(v_i=var_input, episodes_max=10, steps_max=10)
my_plant = simulai.BasePlant(method=my_method, v_i=var_input, v_o=var_output,
                             filename="MaterialHandling.spp", modelname=frame)

# Step 4- Run the simulation

my_plant.process_simulation()
print(my_method.r_episode)
