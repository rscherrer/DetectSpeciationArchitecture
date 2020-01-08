import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
import pandas
import scipy.stats
import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

filenameSimple = "add_data_2.dat"
filenameComplex = "epi_data_2.dat"
ngenes = 400
ntrainsims = 270 # for one set of data (i.e: half)
ntestsims = 30

# read file for simple data
with open(filenameSimple, "rb") as binary_file:
    data = binary_file.read()
all_data = np.frombuffer(data, np.float64)
all_data = all_data[:120000]

# read file for complex data
with open(filenameComplex, "rb") as binary_file:
    data = binary_file.read()
all_data2 =  np.frombuffer(data, np.float64)   


# append complex data to simple
all_data = np.append(all_data, all_data2)

# reshape so each row is one simulation
all_data = np.reshape(all_data, (600,ngenes))


# make label array, first half simple (0), second half complex (1)
all_labels = np.full((600,2), 0)
all_labels[:300 , 0].fill(1)
all_labels[300: , 1].fill(1)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


n_split=10


data = []
test_nodes = [2,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400]
for n in test_nodes:
    print(n)
    scores = np.empty(10)
    for m in range(10):
        total_accuracy = 0
        for train_index,test_index in KFold(n_split).split(all_data):
            x_train,x_test=all_data[train_index],all_data[test_index]
            y_train,y_test=all_labels[train_index],all_labels[test_index]
            model = Sequential()
            model.add(Dense(n, input_dim=400, activation='relu'))
            model.add(Dense(2, activation='softmax'))
            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            model.fit(x_train, y_train,epochs=20, verbose=0)
            (loss, accuracy) = model.evaluate(x_test,y_test)
            total_accuracy = total_accuracy + accuracy
  
        final_accuracy = total_accuracy/10
        scores[m] = final_accuracy
        (mean, lower, upper) = mean_confidence_interval(scores)
    data.append([n,mean,lower,upper,(upper-lower)])

df = pandas.DataFrame(data, columns = ['Nodes', 'Accuracy', 'CI min', 'CI max', 'CI size']) 
df.to_csv('V10_no_PCA.csv')
