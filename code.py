#Import libraries
import numpy as np

#Sets all random values the same
np.random.seed(1)

#Data
x = np.array([[2,9], [1,5], [3,6]])
y = np.array([[92,86,89]]).T

#Creates Random Weights
synaptic_weights = np.random.random((2,1))

#Trains Data 1000 times
for iteration in range(1000):
    #multiplies x values to random weights
    z = np.dot(x, synaptic_weights)
    #Sigmoid calculation
    sigmoid = 1/(1+np.exp(-z))
    #calculation of error
    error = (y - sigmoid)
    #deriviative of sigmoid
    sigmoidDerivative = sigmoid * (1 - sigmoid)
    #adjustment of the wights
    synaptic_weights += np.dot(x.T, error*sigmoidDerivative)


#Final Prediction
study = int(input("Study: "))
sleep = int(input("Sleep: "))
newZ = np.dot(np.array([study,sleep]), synaptic_weights)
print(int(newZ))
