import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
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

centered_data = all_data - all_data.mean(0)
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

reduced_data = pca_reduced_data(20, all_data)

n_split=10
total_accuracy = 0
for train_index,test_index in KFold(n_split).split(reduced_data):
  x_train,x_test=reduced_data[train_index],reduced_data[test_index]
  y_train,y_test=all_labels[train_index],all_labels[test_index]
  model = Sequential()
  model.add(Dense(64, input_dim=20, activation='relu'))
  model.add(Dense(2, activation='softmax'))

  model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

  model.fit(x_train, y_train,epochs=20)
  (loss, accuracy) = model.evaluate(x_test,y_test)
  total_accuracy = total_accuracy + accuracy
  
final_accuracy = total_accuracy/10
print('Accuracy ', final_accuracy)
