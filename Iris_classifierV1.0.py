"""
Created on Mon May 14 18:03:47 2018

@author: abhijeet tripathy

The classifier satifactorily produces an avg effieciency of 96.57% on the
training data set.
"""



import numpy as np
import pandas as pd
#functions
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

#data collection and processing

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
 #shuffles the data along the row
X = dataset.values
np.random.shuffle(X)
data = np.array(X[:,0:4],dtype = np.float64)
traindata = data[0:120,:]   #80% of the data set
testdata = data[120:150,:] #20% of the data set
result = np.array(X[:,4:5], dtype = object)
Yactual_total = np.empty((150,3),dtype = np.int64)
#coverting the actual result to matrix form (multiple output)
for (x,y),value in np.ndenumerate(result):
    if value == 'Iris-setosa':
        Yactual_total[x][0] = 1
        Yactual_total[x][1] = 0
        Yactual_total[x][2] = 0
    elif value == 'Iris-versicolor':
        Yactual_total[x][0] = 0
        Yactual_total[x][1] = 1
        Yactual_total[x][2] = 0
    else:
        Yactual_total[x][0] = 0
        Yactual_total[x][1] = 0
        Yactual_total[x][2] = 1

Yactual_train = Yactual_total[0:120,:]
#variables initialisation
B0 = np.random.rand(1,3)
B1 = np.random.rand(1,3)
W1 = np.random.rand(3,4)
W2 = np.random.rand(3,3)

for i in range(10000):
    #forward propagation
    Z1 = np.array(np.dot(traindata,W1.T), dtype = np.float64) +B0
    A1 = sigmoid(Z1)
    Z2 = np.array(np.dot(A1,W2.T), dtype = np.float64)  + B1
    A2 = sigmoid(Z2)

    #backpropogation
        #for output layer
    dZ2 = A2 -Yactual_total[0:120,:]
    dW2 = np.array(np.dot(dZ2.T,A1),dtype = np.float64)/120
    dB1  = np.sum(dZ2,axis = 0,keepdims = True)/120
        #for the hidden layer
    dZ1 = np.multiply(np.multiply(dZ2,np.sum(W2,axis = 1).T),np.multiply(A1,np.ones((120,3),dtype = np.float64)-A1))
    dW1 = np.array(np.dot(dZ1.T,traindata),dtype = np.float64)/120
    dB0 = np.sum(dZ1,axis=0, keepdims = True)/120

    #gradient descent
    W1 = W1 - 0.5*dW1
    W2 = W2 - 0.5*dW2
    B0 = B0 - 1.25*dB0
    B1 = B1 - 1.25*dB1

    Ytrain = np.copy(A2)
        
#classification of the trained dataset
for (x,y), value in np.ndenumerate(Ytrain):
    if(value<0.5):
        Ytrain[x][y] = 0
    else:
         Ytrain[x][y] = 1
Ytrain = np.array(Ytrain,dtype = np.int64)

#classification of the test datset
#forward propagation
Z1 = np.array(np.dot(testdata,W1.T), dtype = np.float64) +B0
A1 = sigmoid(Z1)
Z2 = np.array(np.dot(A1,W2.T), dtype = np.float64)  + B1
A2 = sigmoid(Z2)
Yexp_test = np.copy(A2)
for (x,y), value in np.ndenumerate(Yexp_test):
    if(value<0.5):
        Yexp_test[x][y] = 0
    else:
         Yexp_test[x][y] = 1
Yexp_test = np.array(Yexp_test,dtype = np.int64)
Yactual_test = Yactual_total[120:150,:]

#calculating the %accuracy of test cases (efficiency)
nomatch = 0
for i in range(30):
        for j in range(3):
            if(Yexp_test[i][j] != Yactual_test[i][j]):
                nomatch+=1
                break
efficiency = (30-nomatch)*3.33
print("The Percentage of accuracy is:-",efficiency)

