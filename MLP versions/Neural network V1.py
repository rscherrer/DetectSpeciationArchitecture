import numpy as np
from keras.models import Sequential
from keras.layers import Dense

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

filenameSimple = "train_data_additive_1.dat"
filenameComplex = "train_data_epistatic_1.dat"
ngenes = 400
ntrainsims = 180 # for one set of data (i.e: half)
ntestsims = 19

# read file for simple data
with open(filenameSimple, "rb") as binary_file:
    data = binary_file.read()
all_data = np.frombuffer(data, np.float64)
train_data = all_data[0:ntrainsims*ngenes]
test_data = all_data[ntrainsims*ngenes:79600]

print("all data", len(all_data))
print("train data", len(train_data))
print("test data", len(test_data))

# read file for complex data
with open(filenameComplex, "rb") as binary_file:
    data = binary_file.read()
all_data2 =  np.frombuffer(data, np.float64)   
train_temp = all_data2[0:ntrainsims*ngenes]
test_temp = all_data2[ntrainsims*ngenes:79600]

print("all data 2", len(all_data2))
print("train temp", len(train_temp))
print("test temp", len(test_temp))

# append complex data to simple
train_data = np.append(train_data, train_temp)
test_data = np.append(test_data, test_temp)
# reshape so each row is one simulation
train_data = np.reshape(train_data, (ntrainsims*2,ngenes))
test_data = np.reshape(test_data, (ntestsims*2,ngenes))

# make label array, first half simple (0), second half complex (1)
train_labels = np.full((ntrainsims * 2,2), 0)
train_labels[:ntrainsims , 0].fill(1)
train_labels[ntrainsims: , 1].fill(1)
print(train_labels)
test_labels = np.full((ntestsims * 2, 2), 0)
test_labels[:ntestsims , 0].fill(1)
test_labels[:ntestsims , 1].fill(1)

model = Sequential()
model.add(Dense(64, input_dim=400, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_data,train_labels,
          epochs=30,
          batch_size=10)

score = model.evaluate(test_data, test_labels, batch_size=20)
print(score)
