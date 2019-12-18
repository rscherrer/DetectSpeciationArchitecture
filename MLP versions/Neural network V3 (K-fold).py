import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold

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
#train_data = all_data[0:ntrainsims*ngenes]
test_data = all_data[:79600]

# read file for complex data
with open(filenameComplex, "rb") as binary_file:
    data = binary_file.read()
all_data2 =  np.frombuffer(data, np.float64)   
#train_temp = all_data2[0:ntrainsims*ngenes]
test_temp = all_data2[:79600]

# append complex data to simple
all_data = np.append(test_data, test_temp)
#test_data = np.append(test_data, test_temp)
# reshape so each row is one simulation
all_data = np.reshape(all_data, (398,ngenes))
#test_data = np.reshape(test_data, (ntestsims*2,ngenes))



# make label array, first half simple (0), second half complex (1)
all_labels = np.full((398,2), 0)
all_labels[:199 , 0].fill(1)
all_labels[199: , 1].fill(1)

n_split=10
total_accuracy = 0
for train_index,test_index in KFold(n_split).split(all_data):
  x_train,x_test=all_data[train_index],all_data[test_index]
  y_train,y_test=all_labels[train_index],all_labels[test_index]
  model = Sequential()
  model.add(Dense(64, input_dim=400, activation='relu'))
  model.add(Dense(2, activation='softmax'))

  model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

  model.fit(x_train, y_train,epochs=20)
  (loss, accuracy) = model.evaluate(x_test,y_test)
  total_accuracy = total_accuracy + accuracy
  
final_accuracy = total_accuracy/10
print('Accuracy ', final_accuracy)

