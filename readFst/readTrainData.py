import numpy as np

# haven't tested any of this!! test on some smaller amounts before using maybe!
# for labels, assumes first half of sims are simple, second half are complex

filenameSimple = "traindata_1_simple.dat"
filenameComplex = "traindata_1_complex.dat"
ngenes = 400
nsims = 400

# read file for simple data
with open(filenameSimple, "rb") as binary_file:
    data = binary_file.read()
train_data = np.frombuffer(data, np.float64)

# read file for complex data
with open(filenameComplex, "rb") as binary_file:
    data = binary_file.read()
temp = np.frombuffer(data, np.float64)

# append complex data to simple
train_data = np.append(train_data, temp)
# reshape so each row is one simulation
train_data = np.reshape(train_data, (nsims,ngenes))

# make label array, first half simple (0), second half complex (1)
train_labels = np.full(ngenes, 0)
train_labels[ngenes/2:].fill(1)