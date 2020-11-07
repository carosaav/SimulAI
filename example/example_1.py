

from simulai import sim

# Step 0 - Definition of model frame

frame = "Modelo"

# Step 1- Definition of input variables

espera = sim.DiscreteVariable(
    "Espera", 60, 300, 10, "Models."+frame+".espera")
stock = sim.DiscreteVariable(
    "Stock", 10, 50, 10, "Models."+frame+".stock")
numviajes = sim.DiscreteVariable(
    "Numero de viajes", 1, 5, 1, "Models."+frame+".numviajes")

var_input = [espera, stock, numviajes]

# Step 2- Definition of output variables

transportes = sim.OutcomeVariable(
    "Distancia Transportes", "Models."+frame+".transportes", 2, 9)
buffers = sim.OutcomeVariable(
    "Llenado buffers", "Models."+frame+".buffers", 3, 20)
salidas = sim.OutcomeVariable(
    "Espera en las Salidas", "Models."+frame+".salidas", 2, 20)

var_output = [transportes, buffers, salidas]

# Step 3- Choice of method, plant and simulation file.
# Enter input variables and output variables.

my_method = sim.Qlearning(v_i=var_input, episodes_max=5, steps_max=10)
my_plant = sim.BasePlant(method=my_method, v_i=var_input, v_o=var_output,
                         filename="MaterialHandling.spp", modelname=frame)

# Step 4- Run the simulation

my_plant.process_simulation()
print(my_method.r_episode)
