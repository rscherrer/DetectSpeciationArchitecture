import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
print(type(train_data))
# make label array, first half simple (0), second half complex (1)
train_labels = np.full((ntrainsims * 2,2), 0)
train_labels[:ntrainsims , 0].fill(1)
train_labels[ntrainsims: , 1].fill(1)
#print(train_labels)
test_labels = np.full((ntestsims * 2, 2), 0)
test_labels[:ntestsims , 0].fill(1)
test_labels[:ntestsims , 1].fill(1)

#Make a random array and then make it positive-definite
centered_data = train_data - train_data.mean(0)
A = np.asmatrix(centered_data.T) * np.asmatrix(centered_data)
U, S, V = np.linalg.svd(A) 
eigvals = S**2 / np.sum(S**2)  # NOTE (@amoeba): These are not PCA eigenvalues. 
                               # This question is about SVD.

fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(400) + 1
plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
#I don't like the default legend so I typically make mine like below, e.g.
#with smaller fonts and a bit transparent so I do not cover up data, and make
#it moveable by the viewer in case upper-right is a bad place for it 
leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                 markerscale=0.4)
leg.get_frame().set_alpha(0.4)
leg.draggable(state=True)
plt.show()

def pca_reduced_data(n,data):
    pca = PCA(n_components=n)
    # subtract mean to center data around origin
    centered_data = data - data.mean(0)
    return pca.fit_transform(centered_data)

reduced_data = pca_reduced_data(20, train_data)
reduced_test = pca_reduced_data(20, test_data)
print(reduced_data)

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(reduced_data,train_labels,
          epochs=20,
          batch_size=20)

score = model.evaluate(reduced_test, test_labels, batch_size=10)
print(score)