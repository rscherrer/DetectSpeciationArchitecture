import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas

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


# read file for complex data
with open(filenameComplex, "rb") as binary_file:
    data = binary_file.read()
all_data2 =  np.frombuffer(data, np.float64)   
train_temp = all_data2[0:ntrainsims*ngenes]
test_temp = all_data2[ntrainsims*ngenes:79600]


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

test_labels = np.full((ntestsims * 2, 2), 0)
test_labels[:ntestsims , 0].fill(1)
test_labels[:ntestsims , 1].fill(1)

data = []
test_nodes = [2,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400]
for n in range(2,401):
    print(n)
    scores = np.empty(10)
    for m in range(10):
        model = Sequential()
        model.add(Dense(n, input_dim=400, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

        model.fit(train_data,train_labels,
          epochs=20,
          batch_size=20, verbose=0)

        (loss,accuracy) = model.evaluate(test_data, test_labels, batch_size=10)
        scores[m] = accuracy

    data.append([n,scores.mean(),np.percentile(scores,95)])

df = pandas.DataFrame(data, columns = ['Nodes', 'Accuracy', 'Percentile']) 
df.to_csv('V1.csv')
