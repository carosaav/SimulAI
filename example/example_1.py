

import simulai


# Step 1- Definition of input variables

espera = simulai.DiscreteVariable(
	"Espera", 60, 300, 10, "Models.Modelo.espera")
stock = simulai.DiscreteVariable(
	"Stock", 10, 50, 10, ".Models.Modelo.stock")
numviajes = simulai.DiscreteVariable(
	"Numero de viajes", 1, 5, 1, ".Models.Modelo.numviajes")

# Step 2- Definition of output variables

transportes = simulai.OutcomeVariable(
	"Distancia Transportes", ".Models.Modelo.transportes", 2, 9)
buffers = simulai.OutcomeVariable(
	"Llenado buffers", ".Models.Modelo.buffers", 3, 20)
salidas = simulai.OutcomeVariable(
	"Espera en las Salidas", ".Models.Modelo.salidas", 2, 20)

# Step 4- Choice of method, plant and simulation file

example1 = simulai.plant_simulation_node(
	m="Q_learning", p= "Plant_1", filename="MaterialHandling.spp")