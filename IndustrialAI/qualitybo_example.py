from BOptimization import opt


# Step 0- Define Images folder path

PATH = opt.StringVariable(
                    "path", "C:/Users/Usuario/Desktop/totalimg/Samples_old")

# Step 1- Define Epochs number

epochs = opt.NumEpochs(50)

# Step 2- Search Space definition

# NumVariable = opt.TypeVariable("Name", LowerLimit, UpperLimit)
# StrVariable = opt.TypeVariable("Name", "value")

lr = opt.ContinuousVariable("Learning_rate", 1e-3, 0.99)
momentum = opt.ContinuousVariable("Momentum", 0.01, 0.99)
bs = opt.DiscreteVariable("Batch_size", 4, 64)
cv1 = opt.DiscreteVariable("Channel_1", 4, 97)
cv2 = opt.DiscreteVariable("Channel_2", 16, 97)
k1 = opt.DiscreteVariable("Kernel_size_1", 2, 16)
k2 = opt.DiscreteVariable("Kernel_size_2", 4, 32)
kp1 = opt.DiscreteVariable("Pool_kernel_size", 2, 24)
optim = opt.DiscreteVariable("Optimizer", 0, 1)  # 0:adam or 1:sgd

var_input = [epochs, lr, momentum, bs, cv1, cv2, k1, k2, kp1, optim, PATH]

# Step 3- Choice of method.

# Enter input variables and output variables.

# my_method = opt.BOgp(v_i=var_input)
my_method = opt.BOtpe(v_i=var_input)


# Step 4- Run the simulation

my_method.boptimization()
