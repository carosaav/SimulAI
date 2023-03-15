from BOptimization import healthopt


# Step 0- Define Images folder path

PATH = healthopt.StringVariable(
    "path", "C:/Users/Usuario/Desktop/totalimg/maper_dataset/42.csv"
)

# Step 1- Define Epochs number

epochs = healthopt.NumEpochs(2)

# Step 2- Search Space definition

# NumVariable = opt.TypeVariable("Name", LowerLimit, UpperLimit)
# StrVariable = opt.TypeVariable("Name", "value")

lr = healthopt.ContinuousVariable("Learning_rate", 1e-3, 0.99)
momentum = healthopt.ContinuousVariable("Momentum", 0.01, 0.99)
bs = healthopt.DiscreteVariable("Kernel_size_1", 2, 16)
lookback = healthopt.DiscreteVariable("Kernel_size_2", 4, 32)
hidden_size = healthopt.DiscreteVariable("Pool_kernel_size", 2, 24)
optim = healthopt.DiscreteVariable("Optimizer", 0, 1)  # 0:adam or 1:sgd

var_input = [epochs, lr, momentum, bs, lookback, hidden_size, optim, PATH]

# Step 3- Choice of method.

# Enter input variables and output variables.

my_method = healthopt.BOgp(v_i=var_input)
# my_method = healthopt.BOtpe(v_i=var_input)

# Step 4- Run the simulation

my_method.boptimization()
