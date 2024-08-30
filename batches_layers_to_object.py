
import numpy as np


#input feature set tends to be denoted as Capital X in machine learning
#it could also be X tran, or X test.

np.random.seed(0)  # seeding the random to 0 for reproductability the " random " numbers that we are using for weights will always be the same
X = [[1,2,3,2.5], 
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

#defining the " hidden layers"

#two ways to initialise a layer: 1,is by using a trained model that you have saved...
# and load that up, this cpuld literally be saving the weights and biases. 
# the second is in the instance of making a new neural network, 
# qst we need to inirialise weights, we will use random values between -1 and 1, 0r even -0.1 and +0.1

# we may have biases as 0, however is we ahave a " dead network "- where the outputs are all 0's then we will change
# the bias to be a non 0 number


#we use the function randn in this - this is a gausian distribution

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #n_inputs is the sampke size, in this example it would be 4. Also * by 0.1 to get rid of any valuies above 1 ( google why you dont want weights above 1 )
        self.biases = np.zeros((1, n_neurons)) #<---1st parameter is a shape, therefore, must be a tuple of the shape
    
    def forward(self, inputs):            
        self.output = np.dot(inputs, self.weights ) + self.biases

    

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)





